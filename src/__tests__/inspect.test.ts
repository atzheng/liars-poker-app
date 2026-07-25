/**
 * inspect.test.ts — "Inspect in Policy Explorer" must reproduce the live game.
 *
 * Clicking inspect on a seat seeds the Explorer with that seat's dealt hand and
 * the move history so far (App.handleInspect), and the Explorer rebuilds every
 * intermediate state from those (trajectory.ts). The policy it shows for each
 * of that seat's decisions must be exactly the policy the agent used in the
 * game — including for a legacy checkpoint, whose observation encodes the hand
 * as RAW ORDERED DIGITS, so the dealt ordering has to survive the round trip
 * through the Explorer's per-digit hand editor.
 */

import { readFileSync } from 'fs';
import { join } from 'path';
import { describe, expect, it } from 'vitest';

import { loadCheckpointBytes } from '../checkpoint';
import { applyAction, isTerminal, newInitialState } from '../game';
import { localPolicySource } from '../policySource';
import { buildTrajectory, handToCounts, resolveHand } from '../trajectory';
import type { GameConfig, GameState, NetworkWeights } from '../types';

function readCheckpoint(path: string) {
  const buf = readFileSync(path);
  const bytes = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return loadCheckpointBytes(bytes as ArrayBuffer);
}

/** Post-deal state with the given hands (bypasses the random deal). */
function dealt(config: GameConfig, hands: number[][]): GameState {
  return {
    ...newInitialState(config),
    hands: hands.map(h => [...h]),
    deal_step: config.total_cards,
  };
}

interface Decision { prefixLen: number; policy: number[] }

/** Play greedily for up to `maxMoves`, recording `seat`'s own decisions. */
async function playGame(
  config: GameConfig,
  weights: NetworkWeights,
  hands: number[][],
  seat: number,
  maxMoves: number,
): Promise<{ moves: number[]; decisions: Decision[] }> {
  const query = localPolicySource(weights, config);
  let state = dealt(config, hands);
  const moves: number[] = [];
  const decisions: Decision[] = [];
  for (let k = 0; k < maxMoves && !isTerminal(state); k++) {
    const { action, policy } = await query(state);
    if (state.current_player === seat) decisions.push({ prefixLen: k, policy });
    moves.push(action);
    state = applyAction(state, config, action);
  }
  return { moves, decisions };
}

/** What App.handleInspect + PolicyExplorer do with that game. */
async function inspect(
  config: GameConfig,
  weights: NetworkWeights,
  hands: number[][],
  seat: number,
  moves: number[],
): Promise<Decision[]> {
  const query = localPolicySource(weights, config);
  const seedHand = [...hands[seat]];                                  // ExplorerInit.hand
  const handCounts = handToCounts(seedHand, config.num_digits);       // hand editor state
  const hand = resolveHand(handCounts, config.hand_length, seedHand); // states built from this
  const { nodes } = buildTrajectory(config, moves, seat, hand);

  const out: Decision[] = [];
  for (const n of nodes.filter(node => node.isActing)) {
    const { policy } = await query(n.state);
    out.push({ prefixLen: n.prefixLen, policy });
  }
  return out;
}

async function expectInspectMatchesGame(
  checkpointPath: string,
  hands: number[][],
  seat: number,
) {
  const { config, weights } = readCheckpoint(checkpointPath);
  const { moves, decisions } = await playGame(config, weights, hands, seat, 6);
  const explorer = await inspect(config, weights, hands, seat, moves);

  // The Explorer's live final node has no played move, so it can hold one extra
  // decision beyond what the truncated game produced.
  expect(explorer.length).toBeGreaterThanOrEqual(decisions.length);
  expect(decisions.length).toBeGreaterThan(0);
  decisions.forEach((d, i) => {
    expect(explorer[i].prefixLen, `node ${i} prefix`).toBe(d.prefixLen);
    for (let a = 0; a < d.policy.length; a++) {
      expect(explorer[i].policy[a], `node ${d.prefixLen} action ${a}`)
        .toBeCloseTo(d.policy[a], 6);
    }
  });
}

describe('inspect → Policy Explorer round trip', () => {
  // Legacy: raw-digit hand, so the dealt ORDER matters. [2,1,3] is deliberately
  // not sorted — expanding it from counts alone would feed the network [1,2,3].
  it('legacy 3x3 preset reproduces the game policies', async () => {
    await expectInspectMatchesGame(
      join(__dirname, '../../public/agents/3x3.msgpack'),
      [[2, 1, 3], [1, 2, 2]],
      0,
    );
  });

  it('legacy 3x3 preset, inspecting the other seat', async () => {
    await expectInspectMatchesGame(
      join(__dirname, '../../public/agents/3x3.msgpack'),
      [[2, 1, 3], [3, 1, 2]],
      1,
    );
  });

  // Current: histogram hand + compact history + max_jump.
  it('factored/compact checkpoint reproduces the game policies', async () => {
    await expectInspectMatchesGame(
      join(__dirname, '../../tests/agents/tiny_factored_compact.msgpack'),
      [[3, 1, 4], [2, 2, 1]],
      0,
    );
  });

  it('mlp/sparse checkpoint reproduces the game policies', async () => {
    await expectInspectMatchesGame(
      join(__dirname, '../../tests/agents/tiny_mlp_sparse.msgpack'),
      [[4, 4, 2], [1, 3, 3]],
      1,
    );
  });
});
