/**
 * trajectory.ts — pure helpers for the Policy Explorer's trajectory inspector.
 *
 * The Explorer reconstructs every intermediate GameState along a move sequence
 * (reusing the engine in game.ts) so it can query the agent's policy at each
 * decision point. These functions are engine-faithful (they only ever transition
 * state via applyAction / legalActionsMask) and side-effect free, so they can be
 * unit-tested against the game engine directly.
 *
 * Hands do NOT affect legality or non-terminal transitions — the engine only
 * reads hands when a challenge RESOLVES (terminal). The agent's observation only
 * reads hands[current_player] (its OWN hand). So we build every state with the
 * acting seat's chosen hand injected and placeholder hands (digit 1) for the
 * other seats; this leaves the acting seat's queried policy exact.
 *
 * These helpers take the acting seat's hand as an ORDERED digit array rather
 * than per-digit counts: the current observation encodes the hand as a
 * histogram (order-irrelevant), but the legacy encoding feeds the raw ordered
 * digits to the network. See resolveHand for how the Explorer's counts editor
 * maps back onto an ordering.
 */

import type { DecodedBid, GameConfig, GameState } from './types';
import {
  applyAction,
  CHALLENGE_ACTION,
  decodeBid,
  isPlayerNode,
  isTerminal,
  legalActionsMask,
  newInitialState,
} from './game';

/** Expand per-digit counts ([count of digit1, ..., count of digitN]) into a hand
 *  array of stored digit values (1-indexed), in ascending digit order. */
export function expandHand(handCounts: number[], handLength: number): number[] {
  const hand: number[] = [];
  handCounts.forEach((c, i) => {
    for (let k = 0; k < c; k++) hand.push(i + 1);
  });
  while (hand.length < handLength) hand.push(1); // pad (only when under-length)
  return hand.slice(0, handLength);
}

/** Invert expandHand: count how many of each digit (1..num_digits) a hand holds. */
export function handToCounts(hand: number[], numDigits: number): number[] {
  const counts = new Array<number>(numDigits).fill(0);
  for (const d of hand) {
    if (d >= 1 && d <= numDigits) counts[d - 1] += 1;
  }
  return counts;
}

/**
 * Turn the Explorer's per-digit counts back into an ordered hand.
 *
 * `seedHand` is the exact hand the Explorer was opened with (from "Inspect in
 * Policy Explorer"). While the counts still describe that hand, its ORDER is
 * preserved — a legacy checkpoint observes the raw ordered digits, so replaying
 * a live game through the Explorer must feed the network the same ordering the
 * game did. Once the user edits the hand there is no ordering to preserve, so
 * the counts are expanded in ascending digit order.
 */
export function resolveHand(
  handCounts: number[],
  handLength: number,
  seedHand?: number[],
): number[] {
  if (seedHand && seedHand.length === handLength) {
    const seedCounts = handToCounts(seedHand, handCounts.length);
    if (seedCounts.every((c, i) => c === handCounts[i])) return [...seedHand];
  }
  return expandHand(handCounts, handLength);
}

/** Post-deal starting state (P0 to open) with the acting seat's hand injected and
 *  placeholder hands (digit 1) for the other seats. `hand` is an ordered digit
 *  array (see resolveHand). */
export function buildInitialState(
  config: GameConfig,
  actingSeat: number,
  hand: number[],
): GameState {
  let state = newInitialState(config);
  const hands = Array.from({ length: config.num_players }, () =>
    new Array<number>(config.hand_length).fill(1),
  );
  hands[actingSeat] = hand.slice(0, config.hand_length);
  state = { ...state, hands, deal_step: config.total_cards };
  return state;
}

/** Replay the sequence to the state reached after `count` moves (default: all).
 *  Stops early if an action is illegal or a terminal/non-player node is reached
 *  (defensive — a sanitized sequence never triggers this). */
export function buildState(
  config: GameConfig,
  sequence: number[],
  actingSeat: number,
  hand: number[],
  count: number = sequence.length,
): GameState {
  let state = buildInitialState(config, actingSeat, hand);
  const n = Math.min(count, sequence.length);
  for (let i = 0; i < n; i++) {
    if (!isPlayerNode(state, config)) break;
    const legal = legalActionsMask(state, config);
    if (!legal[sequence[i]]) break;
    state = applyAction(state, config, sequence[i]);
  }
  return state;
}

/** Return the longest prefix of `raw` that is legal from the start state. Used
 *  after editing/removing a move so the retained continuation stays engine-valid
 *  (trailing moves that become illegal are dropped). Hands/actingSeat never
 *  affect legality, so the result is independent of them — they are passed only
 *  to build the transition state. */
export function sanitizeSequence(
  config: GameConfig,
  actingSeat: number,
  hand: number[],
  raw: number[],
): number[] {
  let state = buildInitialState(config, actingSeat, hand);
  const out: number[] = [];
  for (const a of raw) {
    if (!isPlayerNode(state, config)) break;
    const legal = legalActionsMask(state, config);
    if (!legal[a]) break;
    out.push(a);
    state = applyAction(state, config, a);
  }
  return out;
}

/** Human-readable label for a bid/challenge action. */
export function moveLabel(action: number, config: GameConfig): string {
  if (action === CHALLENGE_ACTION) return 'CHALLENGE';
  const d = decodeBid(action - 1, config);
  return `${d.count} × ${d.number}`;
}

function currentBid(state: GameState, config: GameConfig): DecodedBid | null {
  return state.current_bid_action >= 0 ? decodeBid(state.current_bid_action - 1, config) : null;
}

/** One node along the trajectory: the state at a given prefix and who is to move. */
export interface TimelineNode {
  /** Number of moves already applied to reach this node. */
  prefixLen: number;
  state: GameState;
  /** Seat to move at this node. */
  toMove: number;
  /** True when `toMove` is the acting seat (the seat whose policy we inspect). */
  isActing: boolean;
  /** The move actually played next in the sequence (null for the live final node). */
  takenAction: number | null;
  curBid: DecodedBid | null;
}

export interface Trajectory {
  nodes: TimelineNode[];
  /** Set when the full sequence resolves to a terminal (a challenge counted). */
  terminalState: GameState | null;
}

/**
 * Walk the move sequence, emitting one node per decision point (prefix 0..N). The
 * node at prefix k is the state BEFORE move k; `takenAction` is move k (or null
 * for k === N, the live decision that has not been played yet). Acting-seat nodes
 * are the ones the caller queries the policy for.
 */
export function buildTrajectory(
  config: GameConfig,
  sequence: number[],
  actingSeat: number,
  hand: number[],
): Trajectory {
  let state = buildInitialState(config, actingSeat, hand);
  const nodes: TimelineNode[] = [];
  const n = sequence.length;

  for (let k = 0; k <= n; k++) {
    if (isTerminal(state)) return { nodes, terminalState: state };
    if (!isPlayerNode(state, config)) break;

    const toMove = state.current_player;
    let taken: number | null = null;
    if (k < n) {
      const legal = legalActionsMask(state, config);
      if (!legal[sequence[k]]) {
        // Illegal continuation (never happens for a sanitized sequence). Emit the
        // node without a taken move and stop.
        nodes.push({
          prefixLen: k, state, toMove,
          isActing: toMove === actingSeat, takenAction: null,
          curBid: currentBid(state, config),
        });
        return { nodes, terminalState: null };
      }
      taken = sequence[k];
    }

    nodes.push({
      prefixLen: k, state, toMove,
      isActing: toMove === actingSeat, takenAction: taken,
      curBid: currentBid(state, config),
    });

    if (k < n) state = applyAction(state, config, sequence[k]);
  }

  return { nodes, terminalState: null };
}
