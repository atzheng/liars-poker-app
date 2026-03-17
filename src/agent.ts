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
 */
export function chooseAiAction(
  state: GameState,
  config: GameConfig,
  weights: NetworkWeights,
): number {
  const player = state.current_player;
  const obs    = buildObservation(state, player, config);
  const legal  = legalActionsMask(state, config);
  const policy = networkForward(obs, legal, weights);
  return sampleAction(policy);
}
