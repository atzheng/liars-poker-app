// ---------------------------------------------------------------------------
// Game configuration (mirrors LiarsPokerConfig from liars_poker_jax.py)
// ---------------------------------------------------------------------------

export interface GameConfig {
  num_players: number;
  hand_length: number;
  num_digits: number;
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
  logit: DenseLayer;     // logit/policy head (no activation)
  numLayers: number;     // policy_network_layers.length
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
}
