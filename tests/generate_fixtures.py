#!/usr/bin/env python3
"""Generate test fixtures for the TypeScript game and network implementation.

Run from the liars-poker-app directory:
  python tests/generate_fixtures.py [--checkpoint path/to/agent.msgpack]

Writes tests/fixtures.json with:
  - Deterministic game scenarios (fixed deals, fixed action sequences)
  - Per-step state snapshots, observations, legal masks, returns
  - Network forward-pass tests using the actual checkpoint weights

The TypeScript tests in src/__tests__/ load this file and verify exact
agreement with the Python implementation.
"""

import argparse
import json
import os
import sys

import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OPEN_SPIEL = os.path.join(REPO_ROOT, "repos", "open_spiel")
sys.path.insert(0, OPEN_SPIEL)

from open_spiel.python.games.liars_poker_jax import (  # noqa: E402
    LiarsPokerConfig,
    apply_action,
    new_initial_state,
    is_terminal,
    legal_actions_mask,
    observation_tensor,
    returns,
)

# ────────────────────────────────────────────────────────────────────────────
# Numpy forward pass (mirrors network.ts exactly)
# ────────────────────────────────────────────────────────────────────────────

def matmul_add(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Dense layer: y = x @ kernel + bias  (kernel shape [in, out])."""
    return np.dot(x, kernel) + bias


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def legal_policy(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Masked softmax matching _legal_policy in rnad.py and legalPolicy in network.ts."""
    l_min = logits.min()
    logits = np.where(legal, logits, l_min)
    logits -= logits.max()
    logits *= legal
    exp_logits = np.where(legal, np.exp(logits), 0.0)
    return exp_logits / exp_logits.sum()


def network_forward(obs: np.ndarray, legal: np.ndarray, layers: list) -> tuple:
    """Returns (logits, policy).  layers = list of (kernel, bias) pairs."""
    x = obs.astype(np.float32)
    for kernel, bias in layers[:-1]:   # hidden layers with relu
        x = relu(matmul_add(x, kernel, bias))
    logits = matmul_add(x, layers[-1][0], layers[-1][1])
    policy = legal_policy(logits, legal.astype(np.float32))
    return logits, policy


# ────────────────────────────────────────────────────────────────────────────
# Checkpoint loading (raw msgpack, no flax dependency)
# ────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path: str) -> tuple:
    """Load a Flax msgpack checkpoint and return (config_dict, layers).

    layers is a list of (kernel, bias) pairs in Dense_0 .. Dense_N order,
    where Dense_N is the logit head (not included in hidden count).
    """
    try:
        import flax.serialization
    except ImportError:
        raise ImportError("flax is required to load checkpoints: pip install flax")

    with open(path, "rb") as f:
        state = flax.serialization.msgpack_restore(f.read())

    raw_layers = state["policy_network_layers"]
    # Flax may serialize lists as dicts with int-string keys
    if isinstance(raw_layers, dict):
        raw_layers = [raw_layers[str(k)] for k in sorted(int(k) for k in raw_layers)]
    num_hidden = len(raw_layers)

    params = state["params_target"]["params"]
    layers = []
    for i in range(num_hidden + 1):      # +1 for logit head
        d = params[f"Dense_{i}"]
        kernel = np.array(d["kernel"], dtype=np.float32)   # [in, out]
        bias   = np.array(d["bias"],   dtype=np.float32)   # [out]
        layers.append((kernel, bias))

    jax_cfg = state["jax_config"]
    cfg = {
        "num_players": int(jax_cfg["num_players"]),
        "hand_length": int(jax_cfg["hand_length"]),
        "num_digits":  int(jax_cfg["num_digits"]),
    }
    return cfg, layers


# ────────────────────────────────────────────────────────────────────────────
# Snapshot helpers
# ────────────────────────────────────────────────────────────────────────────

def state_snapshot(state, config: LiarsPokerConfig) -> dict:
    """Serialise a LiarsPokerState to plain Python for JSON output."""
    return {
        "hands":               [[int(v) for v in row] for row in np.array(state.hands)],
        "deal_step":           int(state.deal_step),
        "current_player":      int(state.current_player),
        "bid_originator":      int(state.bid_originator),
        "current_bid_action":  int(state.current_bid_action),
        "num_challenges":      int(state.num_challenges),
        "is_rebid":            bool(state.is_rebid),
        "winner":              int(state.winner),
        "loser":               int(state.loser),
        "bid_history":         [[int(v) for v in row] for row in np.array(state.bid_history)],
        "challenge_history":   [[int(v) for v in row] for row in np.array(state.challenge_history)],
    }


def obs_snapshot(state, config: LiarsPokerConfig) -> dict:
    return {
        str(p): observation_tensor(state, config, p).tolist()
        for p in range(config.num_players)
    }


def legal_snapshot(state, config: LiarsPokerConfig) -> list:
    return legal_actions_mask(state, config).tolist()


def returns_snapshot(state, config: LiarsPokerConfig) -> list:
    return [float(v) for v in returns(state, config)]


def game_step(state, config, action):
    import jax.numpy as jnp
    return apply_action(state, config, jnp.int32(action))


# ────────────────────────────────────────────────────────────────────────────
# Scenario builder
# ────────────────────────────────────────────────────────────────────────────

def build_scenario(config: LiarsPokerConfig, deal_actions: list, action_sequences: list) -> dict:
    """
    deal_actions: chance actions that fix the hands.
    action_sequences: list of {"name": str, "actions": [int, ...]} dicts.
    Returns a fixture dict with the post-deal snapshot and one entry per sequence.
    """
    # Deal phase
    state = new_initial_state(config)
    for a in deal_actions:
        state = game_step(state, config, a)

    result = {
        "deal_actions": deal_actions,
        "post_deal": {
            "state":      state_snapshot(state, config),
            "obs":        obs_snapshot(state, config),
            "legal_mask": legal_snapshot(state, config),
            "returns":    returns_snapshot(state, config),
        },
        "sequences": [],
    }

    for seq in action_sequences:
        s = state
        steps = []
        for a in seq["actions"]:
            s = game_step(s, config, a)
            steps.append({
                "action":     int(a),
                "state":      state_snapshot(s, config),
                "obs":        obs_snapshot(s, config),
                "legal_mask": legal_snapshot(s, config),
                "returns":    returns_snapshot(s, config),
                "terminal":   bool(is_terminal(s)),
            })
        result["sequences"].append({"name": seq["name"], "steps": steps})

    return result


# ────────────────────────────────────────────────────────────────────────────
# Network fixture builder
# ────────────────────────────────────────────────────────────────────────────

def build_network_fixtures(config_dict: dict, layers: list, config: LiarsPokerConfig) -> list:
    """Generate network forward-pass test cases at key game positions."""
    fixtures = []

    def add(name: str, state, player: int):
        obs    = np.array(observation_tensor(state, config, player), dtype=np.float32)
        legal  = np.array(legal_actions_mask(state, config), dtype=np.float32)
        logits, policy = network_forward(obs, legal, layers)
        fixtures.append({
            "name":           name,
            "player":         player,
            "observation":    obs.tolist(),
            "legal_mask":     legal.astype(bool).tolist(),
            "expected_logits": logits.tolist(),
            "expected_policy": policy.tolist(),
        })

    import jax.numpy as jnp

    # Scenario: P0=[1,1,2] P1=[2,0,1], deal=[1,2,1,0,2,1]
    # digit-1 total=3, digit-2 total=2
    deal = [1, 2, 1, 0, 2, 1]
    s = new_initial_state(config)
    for a in deal:
        s = apply_action(s, config, jnp.int32(a))

    # After deal — both players' perspectives
    add("after_deal_p0", s, 0)
    add("after_deal_p1", s, 1)

    # After P0 bids "2x2" (action=5)
    s1 = apply_action(s, config, jnp.int32(5))
    add("after_p0_bid_5_p1_view", s1, 1)

    # After P0 bids "2x2", P1 challenges (action=0)
    s2 = apply_action(s1, config, jnp.int32(0))
    add("after_challenge_p0_view", s2, 0)

    return fixtures


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        metavar="PATH",
        default=None,
        help="Path to .msgpack checkpoint for network forward-pass tests.",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default=os.path.join(os.path.dirname(__file__), "fixtures.json"),
    )
    args = parser.parse_args()

    # Use same config as agent_40000 (2p, 3 cards, 3 digits)
    config = LiarsPokerConfig(num_players=2, hand_length=3, num_digits=3)
    config_dict = {
        "num_players": config.num_players,
        "hand_length":  config.hand_length,
        "num_digits":   config.num_digits,
        "total_cards":  config.total_cards,
        "max_bids":     config.max_bids,
        "num_actions":  config.num_actions,
        "obs_size": (
            config.num_players
            + config.hand_length
            + 2
            + 2 * config.max_bids * config.num_players
        ),
    }

    # ── Game scenarios ────────────────────────────────────────────────────
    # Hands produced by these deal actions:
    #   deal_step 0 → P0[0]=1, 1 → P1[0]=2, 2 → P0[1]=1
    #   deal_step 3 → P1[1]=0, 4 → P0[2]=2, 5 → P1[2]=1
    #   P0=[1,1,2]  P1=[2,0,1]
    # Digit counts (1-indexed bid numbers):
    #   digit-1 (bid_number=1, hand_value=1): P0 has 2, P1 has 1 → total=3
    #   digit-2 (bid_number=2, hand_value=2): P0 has 1, P1 has 1 → total=2
    #   digit-3 (bid_number=3, hand_value=3): impossible (max hand val=2) → total=0
    deal_a = [1, 2, 1, 0, 2, 1]   # P0=[1,1,2], P1=[2,0,1]

    # action encoding for 2p/3-card/3-digit:
    # bid_id = (count-1)*3 + (digit-1);  action = bid_id + 1
    # action 4 = bid_id 3 = "2×1" (count=2, digit=1)  → needs ≥2 digit-1s → total=3 ≥ 2 → P0 WINS
    # action 5 = bid_id 4 = "2×2" (count=2, digit=2)  → total=2 ≥ 2 → P0 WINS
    # action 8 = bid_id 7 = "3×2" (count=3, digit=2)  → total=2 < 3 → P0 LOSES
    # action 3 = bid_id 2 = "1×3" (count=1, digit=3)  → total=0 < 1 → P0 LOSES

    scenarios = [
        build_scenario(config, deal_a, [
            # P0 bids "2×2", P1 challenges, P0 challenges → P0 wins
            {"name": "bidder_wins_2x2",
             "actions": [5, 0, 0]},

            # P0 bids "3×2" → P1 challenges → P0 challenges → P0 loses
            {"name": "bidder_loses_3x2",
             "actions": [8, 0, 0]},

            # P0 bids "1×3" (digit-3 count impossible) → P1 challenges → P0 challenges → P0 loses
            {"name": "bidder_loses_1x3_impossible",
             "actions": [3, 0, 0]},

            # Rebid sequence: P0 bids "2×1" → P1 challenges → P0 rebids "2×2"
            # → P1 challenges again → terminal with "2×2" bid (P0 wins, 2 matches ≥ 2)
            {"name": "rebid_sequence_win",
             "actions": [4, 0, 5, 0]},

            # Rebid sequence ending in loss: P0 bids "2×2" → P1 challenges
            # → P0 rebids "3×2" → P1 challenges → terminal (0 match for digit-3? no...)
            # Actually: "3×2" means count=3, digit=2, total=2 < 3 → P0 loses
            {"name": "rebid_sequence_loss",
             "actions": [5, 0, 8, 0]},
        ]),
    ]

    # Also a scenario where P1 is bid_originator (P0 passes/challenges immediately
    # after P1 bids — but P0 must bid first; there's no "pass")
    # Instead: P0 bids low, P1 raises, P0 challenges → P1 is originator
    # P0 bids "1×1" (action=1), P1 bids "2×1" (action=4), P0 challenges (action=0),
    # P1 challenges (action=0) → terminal with "2×1" bid (count=2, digit-1, total=3 ≥ 2 → P1 wins)
    deal_b = [2, 1, 0, 2, 0, 1]   # P0=[2,0,0], P1=[1,2,1]
    # digit-1 (value=1): P0=0, P1=2 → total=2;  digit-2 (value=2): P0=1, P1=1 → total=2
    scenarios.append(
        build_scenario(config, deal_b, [
            {"name": "p1_is_originator_wins",
             "actions": [1, 4, 0, 0]},  # P0 bids 1×1, P1 raises 2×1, P0 challenges, P1 challenges → P1 bid_orig wins (3-1? wait)
            {"name": "p1_is_originator_loses",
             "actions": [1, 8, 0, 0]},  # P0 bids 1×1, P1 raises 3×2, P0 challenges, P1 challenges → total digit-2=2 < 3 → P1 loses
        ])
    )

    # ── Network fixtures ──────────────────────────────────────────────────
    network_fixtures = None
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # Try the default location relative to the repo root
        candidate = os.path.join(
            os.path.dirname(__file__), "..", "..", "agent_40000.msgpack"
        )
        if os.path.exists(candidate):
            checkpoint_path = candidate

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt_config_dict, layers = load_checkpoint(checkpoint_path)
        # Verify config matches our test config
        assert ckpt_config_dict["num_players"] == config.num_players
        assert ckpt_config_dict["hand_length"]  == config.hand_length
        assert ckpt_config_dict["num_digits"]   == config.num_digits
        network_fixtures = build_network_fixtures(ckpt_config_dict, layers, config)
        print(f"  Generated {len(network_fixtures)} network test cases")
    else:
        print("No checkpoint found — skipping network fixtures")
        print("  Re-run with --checkpoint path/to/agent.msgpack to include them")

    # ── Output ────────────────────────────────────────────────────────────
    output = {
        "meta": {
            "config": config_dict,
            "generated_by": "tests/generate_fixtures.py",
        },
        "scenarios": scenarios,
        "network_tests": network_fixtures,
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
