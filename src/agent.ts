/**
 * agent.ts — Build observation tensor → network forward → sample action.
 *
 * The AI is always player 0.  When it's the AI's turn:
 *   1. Build the observation tensor from the current game state.
 *   2. Build the legal actions boolean array.
 *   3. Run the network forward pass.
 *   4. Sample an action from the resulting policy.
 */

import { buildObservation, legalActionsMask } from './game';
import { networkForward, sampleAction } from './network';
import type { GameConfig, GameState, NetworkWeights } from './types';

/**
 * Choose an action for the AI using the network policy.
 * Uses state.current_player as the acting player (supports multi-player).
 * Returns both the sampled action and the full policy distribution.
 */
export function chooseAiAction(
  state: GameState,
  config: GameConfig,
  weights: NetworkWeights,
  temperature = 1,
  threshold = 0,
): { action: number; policy: number[] } {
  const player = state.current_player;
  const obs    = buildObservation(state, player, config);
  const legal  = legalActionsMask(state, config);
  const policy = networkForward(obs, legal, weights, temperature);

  // Apply threshold: zero out actions below threshold, then renormalize.
  // Always keep at least one action (the highest-probability one) to avoid
  // degenerate cases where all actions are below the threshold.
  let samplingPolicy = policy;
  if (threshold > 0) {
    const thresholded = new Float32Array(policy.length);
    let sum = 0;
    for (let i = 0; i < policy.length; i++) {
      if (policy[i] >= threshold) { thresholded[i] = policy[i]; sum += policy[i]; }
    }
    if (sum > 0) {
      for (let i = 0; i < thresholded.length; i++) thresholded[i] /= sum;
      samplingPolicy = thresholded;
    }
    // if sum === 0 (all below threshold), fall back to original policy
  }

  return { action: sampleAction(samplingPolicy), policy: Array.from(policy) };
}
