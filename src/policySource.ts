/**
 * policySource.ts — one interface for "what does the agent think here?".
 *
 * The Policy Explorer (and anything else that inspects rather than plays) needs
 * a policy for an arbitrary constructed state, which can come from either
 * backend:
 *
 *   server — POST /move to the Python inference server (serve_agent.py /
 *            serve_agent_mlp.py), which runs the real JAX network.
 *   local  — the in-browser forward pass over a loaded checkpoint's weights.
 *
 * Both are exposed as an async `PolicySource` so callers don't branch. The
 * local one resolves immediately; the forward pass is a few hundred µs.
 */

import { buildObservation, legalActionsMask } from './game';
import { networkForwardWithValue } from './network';
import { chooseServerAction } from './serverAgent';
import type { GameConfig, GameState, NetworkWeights } from './types';

export interface PolicyQuery {
  /** Full distribution over actions (0 on illegal actions). */
  policy: number[];
  /** Value head estimate for the acting player, when the backend reports one. */
  value?: number;
  legal?: boolean[];
  /** The greedy (argmax) action — what the agent would play deterministically. */
  action: number;
}

export type PolicySource = (state: GameState) => Promise<PolicyQuery>;

/** Query the Python inference server at `baseUrl`. */
export function serverPolicySource(
  baseUrl: string,
  config: GameConfig,
): PolicySource {
  return async (state: GameState) => {
    // greedy=true so `action` is the argmax rather than a sample.
    const move = await chooseServerAction(baseUrl, state, config, { greedy: true });
    return {
      policy: move.policy,
      value: move.value,
      legal: move.legal,
      action: move.action,
    };
  };
}

/** Run the checkpoint's network in-browser (same path the in-browser AI uses). */
export function localPolicySource(
  weights: NetworkWeights,
  config: GameConfig,
): PolicySource {
  return async (state: GameState) => {
    const observer = state.current_player;
    const obs = buildObservation(state, observer, config);
    const legal = legalActionsMask(state, config);
    const { policy, value } = networkForwardWithValue(obs, legal, weights);

    let action = 0;
    for (let a = 1; a < policy.length; a++) {
      if (policy[a] > policy[action]) action = a;
    }
    return { policy: Array.from(policy), value, legal, action };
  };
}
