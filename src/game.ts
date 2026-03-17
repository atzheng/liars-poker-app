/**
 * game.ts — TypeScript port of liars_poker_jax.py.
 *
 * Key facts:
 *   - Action 0 = CHALLENGE; actions 1..max_bids = BIDs (1-indexed bid_id).
 *   - decode_bid(bid_id): number = bid_id % num_digits + 1,
 *                         count  = floor(bid_id / num_digits) + 1
 *   - max_bids = hand_length * num_digits * num_players
 *   - Chance actions: randint(0, num_digits) → digit value 0..num_digits-1
 *   - bid_history[bid_id][player] = 1 when that player made that bid
 *   - challenge_history[bid_id][player] = 1 when that player challenged bid
 *   - winner/loser = -1 while game not over
 */

import type { GameConfig, GameState, DecodedBid } from './types';

export const CHALLENGE_ACTION = 0;
export const BID_ACTION_OFFSET = 1;

// ---------------------------------------------------------------------------
// State construction
// ---------------------------------------------------------------------------

export function newInitialState(config: GameConfig): GameState {
  const { num_players, hand_length, max_bids } = config;
  return {
    hands: Array.from({ length: num_players }, () => new Array<number>(hand_length).fill(0)),
    deal_step: 0,
    bid_history: Array.from({ length: max_bids }, () => new Array<number>(num_players).fill(0)),
    challenge_history: Array.from({ length: max_bids }, () => new Array<number>(num_players).fill(0)),
    current_player: 0,
    bid_originator: -1,
    current_bid_action: -1,
    num_challenges: 0,
    is_rebid: false,
    winner: -1,
    loser: -1,
  };
}

// ---------------------------------------------------------------------------
// Query functions
// ---------------------------------------------------------------------------

export function isTerminal(state: GameState): boolean {
  return state.winner >= 0 || state.loser >= 0;
}

export function isChanceNode(state: GameState, config: GameConfig): boolean {
  return state.deal_step < config.total_cards;
}

/** Returns true iff the game is in the active betting phase. */
export function isPlayerNode(state: GameState, config: GameConfig): boolean {
  return !isTerminal(state) && !isChanceNode(state, config);
}

// ---------------------------------------------------------------------------
// Bid encoding helpers
// ---------------------------------------------------------------------------

export function decodeBid(bidId: number, config: GameConfig): DecodedBid {
  const number = (bidId % config.num_digits) + 1;
  const count  = Math.floor(bidId / config.num_digits) + 1;
  return { count, number };
}

export function encodeBid(count: number, number: number, config: GameConfig): number {
  return (count - 1) * config.num_digits + (number - 1);
}

// ---------------------------------------------------------------------------
// Legal actions mask
// ---------------------------------------------------------------------------

export function legalActionsMask(state: GameState, config: GameConfig): boolean[] {
  const { num_actions, max_bids, num_players } = config;
  const mask = new Array<boolean>(num_actions).fill(false);

  if (!isPlayerNode(state, config)) return mask;

  // Challenge legal if at least one bid has been made
  const challengeLegal = state.current_bid_action >= 0;

  // Rebid possible when all non-originator players have challenged and no rebid yet
  const isRebidPossible = !state.is_rebid && state.num_challenges === num_players - 1;

  // A player may bid if they are not the originator OR if rebid is possible
  const canBid = state.current_player !== state.bid_originator || isRebidPossible;

  // Bids must be strictly higher than current bid
  const minBidAction = state.current_bid_action < 0
    ? BID_ACTION_OFFSET
    : state.current_bid_action + 1;

  if (challengeLegal) mask[0] = true;

  if (canBid) {
    for (let a = minBidAction; a <= max_bids; a++) {
      mask[a] = true;
    }
  }

  return mask;
}

// ---------------------------------------------------------------------------
// State transition
// ---------------------------------------------------------------------------

export function applyAction(state: GameState, config: GameConfig, action: number): GameState {
  const terminal  = isTerminal(state);
  const chance    = isChanceNode(state, config) && !terminal;
  const notChance = !chance && !terminal;
  const isChallenge = notChance && action === CHALLENGE_ACTION;
  const isBid       = notChance && action !== CHALLENGE_ACTION;

  // Clone arrays (shallow copy sufficient since we replace slices)
  const hands = state.hands.map(row => [...row]);
  const bidHistory = state.bid_history.map(row => [...row]);
  const challengeHistory = state.challenge_history.map(row => [...row]);

  let deal_step = state.deal_step;
  let current_player = state.current_player;
  let bid_originator = state.bid_originator;
  let current_bid_action = state.current_bid_action;
  let num_challenges = state.num_challenges;
  let is_rebid = state.is_rebid;
  let winner = state.winner;
  let loser = state.loser;

  // ------------------------------------------------------------------
  // CHANCE: deal a digit into the current player's hand.
  // deal_step % num_players → receiving player
  // deal_step // num_players → slot
  // ------------------------------------------------------------------
  if (chance) {
    const dealPlayer = deal_step % config.num_players;
    const dealSlot   = Math.floor(deal_step / config.num_players);
    hands[dealPlayer][dealSlot] = action;
    deal_step += 1;
  }

  // ------------------------------------------------------------------
  // CHALLENGE
  // ------------------------------------------------------------------
  if (isChallenge) {
    const challBidIdx = current_bid_action - BID_ACTION_OFFSET;
    challengeHistory[challBidIdx][current_player] += 1;
    num_challenges += 1;

    // Count occurs when all required players have challenged
    const shouldCountNonRebid = !state.is_rebid && num_challenges === config.num_players;
    const shouldCountRebid    = state.is_rebid  && num_challenges === config.num_players - 1;
    const shouldCount         = shouldCountNonRebid || shouldCountRebid;

    if (shouldCount) {
      const bidId    = current_bid_action - BID_ACTION_OFFSET;
      const { count: bidCount, number: bidNumber } = decodeBid(bidId, config);

      // Count matching digits across all hands
      let matches = 0;
      for (const hand of hands) {
        for (const d of hand) if (d === bidNumber) matches++;
      }
      const bidderWins = matches >= bidCount;

      if (bidderWins) winner = bid_originator;
      else            loser  = bid_originator;
    }
  }

  // ------------------------------------------------------------------
  // BID
  // ------------------------------------------------------------------
  if (isBid) {
    const safeBidIdx = action - BID_ACTION_OFFSET;
    bidHistory[safeBidIdx][current_player] += 1;
    current_bid_action = action;
    is_rebid = current_player === state.bid_originator;
    bid_originator = current_player;
    num_challenges = 0;
  }

  // Advance current player (always, for all action types)
  current_player = (current_player + 1) % config.num_players;

  return {
    hands,
    deal_step,
    bid_history: bidHistory,
    challenge_history: challengeHistory,
    current_player,
    bid_originator,
    current_bid_action,
    num_challenges,
    is_rebid,
    winner,
    loser,
  };
}

// ---------------------------------------------------------------------------
// Deal an entire game at once (for UI convenience)
// ---------------------------------------------------------------------------

/** Apply all chance actions in one go, using random digits. */
export function dealGame(config: GameConfig): GameState {
  let state = newInitialState(config);
  while (isChanceNode(state, config)) {
    const digit = Math.floor(Math.random() * config.num_digits);
    state = applyAction(state, config, digit);
  }
  return state;
}

// ---------------------------------------------------------------------------
// Observation tensor (mirrors observation_tensor in liars_poker_jax.py)
// ---------------------------------------------------------------------------

export function buildObservation(
  state: GameState,
  player: number,
  config: GameConfig,
): Float32Array {
  const { num_players, hand_length, max_bids } = config;
  const obs: number[] = [];

  // player one-hot [num_players]
  for (let p = 0; p < num_players; p++) obs.push(p === player ? 1 : 0);

  // private hand [hand_length] (zeros during deal)
  const dealingDone = state.deal_step >= config.total_cards;
  for (let i = 0; i < hand_length; i++) {
    obs.push(dealingDone ? state.hands[player][i] : 0);
  }

  // is_rebid [1]
  obs.push(state.is_rebid ? 1 : 0);

  // is_terminal [1]
  obs.push(isTerminal(state) ? 1 : 0);

  // bid_history [max_bids * num_players] — raveled row-major
  for (let b = 0; b < max_bids; b++) {
    for (let p = 0; p < num_players; p++) obs.push(state.bid_history[b][p]);
  }

  // challenge_history [max_bids * num_players]
  for (let b = 0; b < max_bids; b++) {
    for (let p = 0; p < num_players; p++) obs.push(state.challenge_history[b][p]);
  }

  return new Float32Array(obs);
}

// ---------------------------------------------------------------------------
// Returns
// ---------------------------------------------------------------------------

export function getReturns(state: GameState, config: GameConfig): number[] {
  const { num_players, bid_originator } = { ...state, ...config };
  const result = new Array<number>(config.num_players).fill(0);

  if (state.winner >= 0) {
    // Bidder wins: bidder gets +(num_players-1), others get -1
    for (let p = 0; p < config.num_players; p++) {
      result[p] = p === bid_originator ? config.num_players - 1 : -1;
    }
  } else if (state.loser >= 0) {
    // Bidder loses: bidder gets -(num_players-1), others get +1
    for (let p = 0; p < config.num_players; p++) {
      result[p] = p === bid_originator ? -(config.num_players - 1) : 1;
    }
  }
  return result;
}
