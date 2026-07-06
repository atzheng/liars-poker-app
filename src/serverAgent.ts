/**
 * serverAgent.ts — client for the Python transformer inference backend.
 *
 * Instead of running MLP inference in-browser (network.ts), "server mode" posts
 * the current game state to a small Python server (src/serve_agent.py in the
 * liars-poker-open-spiel repo) that runs the REAL trained transformer (lpt) and
 * returns the policy + chosen action. This keeps the tested attention / 2D-RoPE
 * / equivariant-head code on the Python side rather than re-porting it to TS.
 *
 * The server reconstructs a full LiarsPokerState from the posted GameState (the
 * app's GameState is a 1:1 port), so we simply serialize the state fields.
 * The AI observes from state.current_player (matches the app: the AI moves
 * whenever it is not the human's turn).
 */

import type { GameConfig, GameState } from './types';
import { buildGameConfig } from './checkpoint';

export interface ServerInfo {
  config: GameConfig;
  network_type: string;
  checkpoint: string;
  history_encoding: string;
}

/** GET {baseUrl}/config → game dims from the loaded checkpoint. */
export async function fetchServerConfig(baseUrl: string): Promise<ServerInfo> {
  const url = baseUrl.replace(/\/+$/, '') + '/config';
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET /config → HTTP ${res.status}`);
  const j = await res.json() as {
    num_players: number; hand_length: number; num_digits: number;
    network_type: string; checkpoint: string; history_encoding: string;
  };
  return {
    config: buildGameConfig(j.num_players, j.hand_length, j.num_digits),
    network_type: j.network_type,
    checkpoint: j.checkpoint,
    history_encoding: j.history_encoding,
  };
}

export interface ServerMove {
  action: number;
  policy: number[];
  /** Value head estimate for the acting player (present in newer servers). */
  value?: number;
  /** Legal-action mask as computed server-side (present in newer servers). */
  legal?: boolean[];
  /** Seat the server observed from (== current_player). */
  observer?: number;
}

/**
 * POST {baseUrl}/move with the current game state; returns the agent's chosen
 * action and full policy. `greedy` picks argmax; otherwise the server samples
 * from the policy re-tempered by `temperature`.
 */
export async function chooseServerAction(
  baseUrl: string,
  state: GameState,
  _config: GameConfig,
  opts: { temperature?: number; greedy?: boolean; threshold?: number } = {},
): Promise<ServerMove> {
  const url = baseUrl.replace(/\/+$/, '') + '/move';
  const payload = {
    hands: state.hands,
    deal_step: state.deal_step,
    bid_history: state.bid_history,
    challenge_history: state.challenge_history,
    current_player: state.current_player,
    bid_originator: state.bid_originator,
    current_bid_action: state.current_bid_action,
    num_challenges: state.num_challenges,
    is_rebid: state.is_rebid,
    winner: state.winner,
    loser: state.loser,
    temperature: opts.temperature ?? 1,
    greedy: opts.greedy ?? false,
    threshold: opts.threshold ?? 0,
  };
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = '';
    try { detail = (await res.json())?.error ?? ''; } catch { /* ignore */ }
    throw new Error(`POST /move → HTTP ${res.status}${detail ? `: ${detail}` : ''}`);
  }
  const j = await res.json() as ServerMove;
  return j;
}
