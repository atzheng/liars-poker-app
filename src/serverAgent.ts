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
  /** false → server is up but no checkpoint is loaded yet (choose one via /load). */
  loaded: boolean;
  /** Present only when `loaded` is true. */
  config?: GameConfig;
  network_type?: string;
  checkpoint?: string;
  /** Full path/uri of the loaded checkpoint (for display / re-selection). */
  checkpointPath?: string;
  history_encoding?: string;
  /** Action-space abstraction bound from the checkpoint (null = unrestricted). */
  maxJump: number | null;
  /** Directory / s3 prefix the server suggests browsing for checkpoints. */
  defaultDir?: string | null;
}

/** Raw JSON returned by the server's /config and /load endpoints. */
interface RawConfig {
  loaded?: boolean;
  num_players?: number; hand_length?: number; num_digits?: number;
  network_type?: string; checkpoint?: string; checkpoint_path?: string;
  history_encoding?: string;
  // max_jump present on the factored_mlp (maxjump) backend; absent => null.
  max_jump?: number | null;
  // jb/gpu-abstraction backend also serves the opening base count + count cap.
  first_bid_base_count?: number | null;
  max_bid_count?: number | null;
  // present on the not-yet-loaded /config response.
  default_dir?: string | null;
}

/** Parse a /config or /load JSON body into a ServerInfo. */
function parseServerConfig(j: RawConfig): ServerInfo {
  // The mlp backend now returns {loaded: false} before any checkpoint is
  // loaded. Older backends always have a checkpoint, so treat a missing
  // `loaded` flag but present dims as loaded.
  const loaded = j.loaded ?? (j.num_players != null);
  if (!loaded) {
    return { loaded: false, maxJump: null, defaultDir: j.default_dir ?? null };
  }
  const maxJump = j.max_jump ?? null;
  const firstBidBaseCount = j.first_bid_base_count ?? null;
  const maxBidCount = j.max_bid_count ?? null;
  return {
    loaded: true,
    config: buildGameConfig(j.num_players!, j.hand_length!, j.num_digits!, {
      maxJump, firstBidBaseCount, maxBidCount,
      // The server builds observations itself, but keep the config honest so
      // anything computing an observation locally matches the backend.
      handEncoding: 'histogram',
      historyEncoding: j.history_encoding === 'compact' ? 'compact' : 'sparse',
    }),
    network_type: j.network_type,
    checkpoint: j.checkpoint,
    checkpointPath: j.checkpoint_path,
    history_encoding: j.history_encoding,
    maxJump,
    defaultDir: j.default_dir ?? null,
  };
}

/** GET {baseUrl}/config → game dims from the loaded checkpoint (if any). */
export async function fetchServerConfig(baseUrl: string): Promise<ServerInfo> {
  const url = baseUrl.replace(/\/+$/, '') + '/config';
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET /config → HTTP ${res.status}`);
  return parseServerConfig(await res.json() as RawConfig);
}

/**
 * GET {baseUrl}/checkpoints → list of `*.msgpack` checkpoints the server can
 * load. `dir` overrides the server's default browse directory / s3 prefix.
 * Only the factored_mlp backend implements this; older backends 404 → [].
 */
export async function fetchCheckpoints(
  baseUrl: string,
  dir?: string,
): Promise<{ dir: string | null; checkpoints: string[] }> {
  let url = baseUrl.replace(/\/+$/, '') + '/checkpoints';
  if (dir) url += `?dir=${encodeURIComponent(dir)}`;
  const res = await fetch(url);
  if (res.status === 404) return { dir: dir ?? null, checkpoints: [] };
  const j = await res.json() as { dir?: string | null; checkpoints?: string[]; error?: string };
  if (!res.ok) throw new Error(`GET /checkpoints → HTTP ${res.status}${j.error ? `: ${j.error}` : ''}`);
  return { dir: j.dir ?? null, checkpoints: j.checkpoints ?? [] };
}

/**
 * POST {baseUrl}/load {checkpoint} → load a checkpoint LIVE and swap it in.
 * Returns the newly-loaded server config (same shape as /config).
 */
export async function loadServerCheckpoint(
  baseUrl: string,
  checkpoint: string,
): Promise<ServerInfo> {
  const url = baseUrl.replace(/\/+$/, '') + '/load';
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ checkpoint }),
  });
  const j = await res.json() as RawConfig & { error?: string };
  if (!res.ok) throw new Error(`POST /load → HTTP ${res.status}${j.error ? `: ${j.error}` : ''}`);
  return parseServerConfig(j);
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
