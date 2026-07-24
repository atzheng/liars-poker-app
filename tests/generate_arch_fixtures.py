#!/usr/bin/env python3
"""Generate tiny checkpoints + golden inference data for the app's arch tests.

The app must load BOTH generations of Liar's Poker checkpoints:

  legacy   flat `mlp` head, raw-digit hand, sparse history, no metadata
  current  `mlp` or `factored_mlp` head, per-digit histogram hand, sparse or
           compact history, optional `max_jump`

Real checkpoints are 20 MB+, so this script builds miniature ones (random
weights, a few-unit MLP) with the real Flax modules from the training repo, and
records what the real JAX forward pass produces for a handful of random states.
`src/__tests__/checkpoint.test.ts` replays those states through the TypeScript
loader/observation/forward pass and asserts agreement.

Usage:
  python tests/generate_arch_fixtures.py \
      --repo ../liars-poker-open-spiel \
      --python ../liars-poker-open-spiel/.venv/bin/python   # (just run it with that python)

Writes tests/agents/tiny_*.msgpack and tests/arch_fixtures.json.
"""

import argparse
import json
import os
import sys

import numpy as np


def build(repo: str, out_dir: str, fixtures_path: str) -> None:
    sys.path.insert(0, os.path.join(repo, "src"))

    import flax
    import jax
    import jax.numpy as jnp

    import liars_poker_jax as lp
    from rnad import EnvStep
    from train_jax import make_network

    # Each variant pins one combination of head / history encoding / max_jump.
    variants = [
        dict(name="tiny_factored_compact", network_type="factored_mlp",
             history_encoding="compact", max_jump=2, layers=[16, 12], seed=1),
        dict(name="tiny_mlp_sparse", network_type="mlp",
             history_encoding="sparse", max_jump=None, layers=[16, 12], seed=2),
    ]
    num_players, hand_length, num_digits = 2, 3, 4

    fixtures = {"variants": []}
    os.makedirs(out_dir, exist_ok=True)

    for v in variants:
        cfg = lp.LiarsPokerConfig(
            num_players=num_players, hand_length=hand_length,
            num_digits=num_digits, max_jump=v["max_jump"],
        )
        network = make_network(v["network_type"], cfg, v["layers"])
        obs_size = lp.observation_size(
            cfg, history_encoding=v["history_encoding"])
        dummy = EnvStep(
            obs=jnp.zeros(obs_size, dtype=jnp.float32),
            legal=jnp.ones(cfg.num_actions, dtype=jnp.int8),
            player_id=jnp.int32(0),
            valid=jnp.float32(1.0),
            rewards=jnp.zeros(cfg.num_players),
        )
        # Fixed seed per variant: regenerating must reproduce the same weights.
        params = network.init(jax.random.PRNGKey(v["seed"]), dummy)

        state_dict = {
            "params_target": params,
            "jax_config": {
                "num_players": num_players,
                "hand_length": hand_length,
                "num_digits": num_digits,
            },
            "policy_network_layers": list(v["layers"]),
            "network_type": v["network_type"],
            "history_encoding": v["history_encoding"],
        }
        if v["max_jump"] is not None:
            state_dict["jax_config"]["max_jump"] = int(v["max_jump"])

        ckpt_path = os.path.join(out_dir, f"{v['name']}.msgpack")
        with open(ckpt_path, "wb") as f:
            f.write(flax.serialization.msgpack_serialize(
                flax.serialization.to_state_dict(state_dict)))

        # --- golden forward passes over random reachable states -------------
        def forward(obs, legal):
            env = EnvStep(obs=obs, legal=legal, player_id=jnp.int32(0),
                          valid=jnp.float32(1.0),
                          rewards=jnp.zeros(cfg.num_players))
            pi, v, _log_pi, _logit = network.apply(params, env)
            return pi, v

        rng = np.random.RandomState(7)
        cases = []
        while len(cases) < 12:
            state = lp.new_initial_state(cfg)
            while bool(lp.is_chance_node(state, cfg)):
                state = lp.apply_action(
                    state, cfg, jnp.int32(int(rng.randint(cfg.num_digits))))
            steps = 0
            while not bool(lp.is_terminal(state)) and steps < 12:
                legal = np.asarray(
                    lp.legal_actions_mask(state, cfg)).astype(bool)
                if not legal.any():
                    break
                observer = int(state.current_player)
                obs = lp.observation_tensor(
                    state, cfg, jnp.int32(observer),
                    history_encoding=v["history_encoding"])
                legal_arr = lp.legal_actions_mask(state, cfg)
                pi, value = forward(obs, legal_arr)
                cases.append({
                    "state": {
                        "hands": np.asarray(state.hands).tolist(),
                        "deal_step": int(state.deal_step),
                        "bid_history": np.asarray(state.bid_history).tolist(),
                        "challenge_history":
                            np.asarray(state.challenge_history).tolist(),
                        "current_player": int(state.current_player),
                        "bid_originator": int(state.bid_originator),
                        "current_bid_action": int(state.current_bid_action),
                        "num_challenges": int(state.num_challenges),
                        "is_rebid": bool(state.is_rebid),
                        "winner": int(state.winner),
                        "loser": int(state.loser),
                    },
                    "observer": observer,
                    "obs": np.asarray(obs, dtype=np.float64).tolist(),
                    "legal": legal.tolist(),
                    "policy": np.asarray(pi, dtype=np.float64).ravel().tolist(),
                    "value": float(np.asarray(value).ravel()[0]),
                })
                choices = np.flatnonzero(legal)
                weights = np.where(choices == 0, 0.2, 1.0)
                action = int(rng.choice(choices, p=weights / weights.sum()))
                state = lp.apply_action(state, cfg, jnp.int32(action))
                steps += 1
                if len(cases) >= 12:
                    break

        fixtures["variants"].append({
            "name": v["name"],
            "checkpoint": os.path.relpath(ckpt_path,
                                          os.path.dirname(fixtures_path)),
            "network_type": v["network_type"],
            "history_encoding": v["history_encoding"],
            "hand_encoding": "histogram",
            "max_jump": v["max_jump"],
            "num_players": num_players,
            "hand_length": hand_length,
            "num_digits": num_digits,
            "num_actions": cfg.num_actions,
            "obs_size": obs_size,
            "cases": cases,
        })
        print(f"{v['name']}: obs_size={obs_size} actions={cfg.num_actions} "
              f"cases={len(cases)} -> {ckpt_path}")

    with open(fixtures_path, "w") as f:
        json.dump(fixtures, f)
    print(f"wrote {fixtures_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo", default=os.path.join(here, "..", "..",
                                       "liars-poker-open-spiel"),
        help="path to the liars-poker-open-spiel checkout (for src/*.py)")
    ap.add_argument("--out-dir", default=os.path.join(here, "agents"))
    ap.add_argument("--fixtures",
                    default=os.path.join(here, "arch_fixtures.json"))
    args = ap.parse_args()
    build(args.repo, args.out_dir, args.fixtures)


if __name__ == "__main__":
    main()
