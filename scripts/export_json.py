#!/usr/bin/env python3
"""Export a Liar's Poker checkpoint to JSON for use with the web app.

Accepts either:
  - A flax msgpack checkpoint (.msgpack) produced by convert_checkpoint.py
  - An old haiku pickle checkpoint (.pickle) directly

Usage:
  python export_json.py checkpoint.msgpack [-o out.json]
  python export_json.py agent_300.pickle [-o out.json]

The output JSON has this structure:
  {
    "jax_config": {"num_players": 2, "hand_length": 5, "num_digits": 5},
    "policy_network_layers": [256, 256, 256],
    "params_target": {
      "Dense_0": {"kernel": [[...], ...], "bias": [...]},
      "Dense_1": ...,
      ...
    }
  }
"""

import argparse
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Load from msgpack (flax serialization format)
# ---------------------------------------------------------------------------

def load_msgpack(path: str) -> dict:
    import flax.serialization
    with open(path, 'rb') as f:
        data = f.read()
    return flax.serialization.msgpack_restore(data)


# ---------------------------------------------------------------------------
# Load from old haiku pickle, then convert
# ---------------------------------------------------------------------------

def load_pickle_and_convert(path: str) -> dict:
    import pickle
    import jax
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'liars-poker-open-spiel', 'src'))

    class _OldAgentData:
        def __setstate__(self, state):
            self.__dict__.update(state)

    class _AgentUnpickler(pickle.Unpickler):
        _REDIRECT = frozenset({'JaxLiarsPokerRNaDSolver', 'RNaDSolver', 'ParallelRNaDSolver'})
        def find_class(self, module, name):
            if name in self._REDIRECT:
                return _OldAgentData
            return super().find_class(module, name)

    with open(path, 'rb') as f:
        old = _AgentUnpickler(f).load()

    from open_spiel.python.algorithms.rnad.rnad import RNaDConfig
    from open_spiel.python.games.liars_poker_jax import LiarsPokerConfig
    from train_jax import JaxLiarsPokerRNaDSolver

    jax_config = old.jax_config
    rnad_config = old.config
    agent = JaxLiarsPokerRNaDSolver(rnad_config, jax_config)

    def _remap_tree(old_tree, new_template):
        new_treedef = jax.tree_util.tree_structure(new_template)
        old_leaves = jax.tree_util.tree_leaves(old_tree)
        return jax.tree_util.tree_unflatten(new_treedef, old_leaves)

    agent.params_target = _remap_tree(old.params_target, agent.params_target)

    state = {
        'params_target': agent.params_target,
        'jax_config': {
            'num_players': jax_config.num_players,
            'hand_length': jax_config.hand_length,
            'num_digits': jax_config.num_digits,
        },
        'policy_network_layers': list(rnad_config.policy_network_layers),
    }
    return state


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def arrays_to_lists(obj):
    """Recursively convert numpy arrays to Python lists."""
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if hasattr(obj, '__jax_array__') or type(obj).__name__ == 'Array':
        return np.array(obj).tolist()
    if isinstance(obj, dict):
        return {k: arrays_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [arrays_to_lists(x) for x in obj]
    return obj


def export(state: dict, out_path: str) -> None:
    layers = state['policy_network_layers']
    num_hidden = len(layers)

    params = state['params_target']['params']
    params_out = {}
    for i in range(num_hidden + 1):  # +1 for logit head
        key = f'Dense_{i}'
        layer = params[key]
        params_out[key] = {
            'kernel': arrays_to_lists(layer['kernel']),
            'bias':   arrays_to_lists(layer['bias']),
        }

    jax_cfg = state['jax_config']
    out = {
        'jax_config': {
            'num_players': int(jax_cfg['num_players']),
            'hand_length': int(jax_cfg['hand_length']),
            'num_digits':  int(jax_cfg['num_digits']),
        },
        'policy_network_layers': [int(x) for x in layers],
        'params_target': params_out,
    }

    with open(out_path, 'w') as f:
        json.dump(out, f)
    print(f'Saved {out_path}')

    # Print size info
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'  File size: {size_mb:.1f} MB')
    cfg = out['jax_config']
    print(f'  Config: {cfg["num_players"]}p {cfg["hand_length"]}cards {cfg["num_digits"]}digits')
    print(f'  Layers: {layers}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='checkpoint file (.msgpack or .pickle)')
    parser.add_argument('-o', '--output', help='output JSON path (default: <input>.json)')
    args = parser.parse_args()

    out_path = args.output or os.path.splitext(args.input)[0] + '.json'

    if args.input.endswith('.pickle'):
        print(f'Loading pickle: {args.input}')
        state = load_pickle_and_convert(args.input)
    else:
        print(f'Loading msgpack: {args.input}')
        state = load_msgpack(args.input)

    export(state, out_path)


if __name__ == '__main__':
    main()
