// ---------------------------------------------------------------------------
// Observation encodings (mirror the flags in liars_poker_jax.observation_tensor)
// ---------------------------------------------------------------------------

/** How the observer's own hand is encoded.
 *  'digits'    — the raw dealt digits, one slot per card [hand_length]. The
 *                original encoding; used by checkpoints trained before the
 *                histogram port (they carry no `network_type` metadata).
 *  'histogram' — per-digit counts [num_digits]. The current encoding: the hand
 *                is order-irrelevant, so counting collapses up to num_digits!
 *                equivalent hands into one observation. */
export type HandEncoding = 'digits' | 'histogram';

/** How the bid/challenge history is encoded.
 *  'sparse'  — the two raw [max_bids, num_players] matrices, raveled.
 *  'compact' — a dense structured summary of length 3*num_digits +
 *              num_players + 5 (see buildCompactHistory in game.ts). */
export type HistoryEncoding = 'sparse' | 'compact';

/** Policy/value architecture (checkpoint field `network_type`).
 *  'mlp'          — flat Dense(num_actions) logit head.
 *  'factored_mlp' — logits from a dot product between a per-state query and
 *                   per-action keys derived from a static action-feature table.
 *  Transformer ('lpt') checkpoints are server-only. */
export type NetworkType = 'mlp' | 'factored_mlp';

// ---------------------------------------------------------------------------
// Game configuration (mirrors LiarsPokerConfig from liars_poker_jax.py)
// ---------------------------------------------------------------------------

export interface GameConfig {
  num_players: number;
  hand_length: number;
  num_digits: number;
  // Optional action-space abstraction (max_jump): the next bid's COUNT is
  // restricted to at most `max_jump` above the current bid's count (base_count
  // = current bid count, or 1 for the opening). undefined/null => unrestricted.
  // Mirrors LiarsPokerConfig.max_jump; served on the backend's GET /config.
  maxJump?: number | null;
  // jb/gpu-abstraction game params (served on GET /config). firstBidBaseCount is
  // the lowest COUNT offered on the opening bid (jb default 2; this agent 1);
  // maxBidCount caps the absolute bid COUNT (jb ``max_bid_count``). undefined =>
  // opening base 1 / no count cap (the legacy in-browser behaviour).
  firstBidBaseCount?: number | null;
  maxBidCount?: number | null;
  // Observation layout the agent was trained with. Both default to the legacy
  // encoding ('digits'/'sparse') when absent, so configs coming from older
  // checkpoints (and the committed fixtures) keep their original observation.
  handEncoding?: HandEncoding;
  historyEncoding?: HistoryEncoding;
  // derived
  total_cards: number;   // num_players * hand_length
  max_bids: number;      // hand_length * num_digits * num_players
  num_actions: number;   // max_bids + 1
  obs_size: number;
}

// ---------------------------------------------------------------------------
// Network weights (from params_target.params in the checkpoint)
// ---------------------------------------------------------------------------

export interface DenseLayer {
  kernel: Float32Array;  // flat C-order, shape [in_size, out_size]
  bias: Float32Array;    // shape [out_size]
  inSize: number;
  outSize: number;
}

export interface NetworkWeights {
  hidden: DenseLayer[];  // relu-activated hidden layers
  // Logit/policy head (no activation), applied to the last hidden activation.
  // For a factored_mlp checkpoint this is the FOLDED equivalent of the
  // query/key head — the per-action keys are static, so the whole head
  // collapses into one Dense layer at load time (see foldFactoredHead).
  logit: DenseLayer;
  // Scalar value head (also applied to the last hidden activation). Only the
  // Policy Explorer reads it, so it is optional — a checkpoint without one
  // still plays.
  value?: DenseLayer;
  numLayers: number;     // policy_network_layers.length
  // Architecture the weights came from (display/diagnostics only — inference
  // is identical once the factored head has been folded).
  networkType?: NetworkType;
}

// ---------------------------------------------------------------------------
// Game state (mirrors LiarsPokerState from liars_poker_jax.py)
// ---------------------------------------------------------------------------

export interface GameState {
  hands: number[][];           // [num_players][hand_length]
  deal_step: number;
  bid_history: number[][];     // [max_bids][num_players] — 1 where player bid
  challenge_history: number[][];
  current_player: number;
  bid_originator: number;      // -1 if no bid yet
  current_bid_action: number;  // -1 if no bid yet
  num_challenges: number;
  is_rebid: boolean;
  winner: number;              // -1 if not terminal
  loser: number;               // -1 if not terminal
}

// ---------------------------------------------------------------------------
// Bid display helper
// ---------------------------------------------------------------------------

export interface DecodedBid {
  count: number;   // how many (1-indexed)
  number: number;  // which digit (1-indexed)
}

// ---------------------------------------------------------------------------
// History entry for the bid log
// ---------------------------------------------------------------------------

export type HistoryEntryType = 'bid' | 'challenge' | 'deal';

export interface HistoryEntry {
  type: HistoryEntryType;
  player: number;
  action: number;       // raw action id
  decodedBid?: DecodedBid;
  label: string;
  policy?: number[];    // AI policy distribution at the time of action
}
