/**
 * PolicyExplorer.tsx — analysis/debugging tool (NOT gameplay).
 *
 * Lets the user construct an arbitrary decision node by (1) choosing the acting
 * player's HAND (how many of each digit it holds) and (2) building a SEQUENCE OF
 * MOVES from a fresh deal, then queries the trained transformer (via /move) for
 * its action distribution at that state and renders it with PolicyHeatmap.
 *
 * How the state is built (reuses src/game.ts, the port of liars_poker_jax):
 *   - Start from a post-deal state (deal_step = total_cards, P0 to open).
 *   - The move sequence is applied with applyAction(); the builder only offers
 *     moves permitted by legalActionsMask() (strictly-increasing bids, challenge
 *     only after a bid, rebid rules, ...), so every constructed state is valid.
 *   - Hands do NOT affect legality or (non-terminal) transitions — the engine
 *     only reads hands when a challenge RESOLVES (terminal). So we build the
 *     sequence with placeholder opponent hands and inject the user's chosen hand
 *     into the acting seat at query time. The transformer's structured_observation
 *     only ever reads hands[current_player], i.e. its OWN hand, so the opponent's
 *     hand is irrelevant to the queried policy (kept hidden/arbitrary).
 *
 * The query is the policy of whoever is TO MOVE (the acting seat). We only query
 * when the constructed state is a non-terminal decision node AND it is the acting
 * seat's turn; otherwise we prompt the user to add the intervening move(s).
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import type { GameConfig, GameState } from '../types';
import {
  applyAction,
  CHALLENGE_ACTION,
  decodeBid,
  isPlayerNode,
  isTerminal,
  legalActionsMask,
  newInitialState,
} from '../game';
import { chooseServerAction } from '../serverAgent';
import PolicyHeatmap from './PolicyHeatmap';

interface Props {
  config: GameConfig;
  serverUrl: string;
  serverLabel?: string;
  onBack: () => void;
}

interface QueryResult {
  policy: number[];
  value?: number;
  legal?: boolean[];
  action: number;
}

/** Expand per-digit counts ([count of digit1, ..., count of digitN]) into a
 *  hand array of stored digit values (1-indexed). Order is irrelevant to the
 *  histogram-based model. */
function expandHand(handCounts: number[], handLength: number): number[] {
  const hand: number[] = [];
  handCounts.forEach((c, i) => {
    for (let k = 0; k < c; k++) hand.push(i + 1);
  });
  while (hand.length < handLength) hand.push(1); // pad (only when under-length)
  return hand.slice(0, handLength);
}

/** Build a post-deal state, inject the acting seat's hand, replay the sequence.
 *  Placeholder hands (digit 1) are used for the non-acting seats. */
function buildState(
  config: GameConfig,
  sequence: number[],
  actingSeat: number,
  handCounts: number[],
): GameState {
  let state = newInitialState(config);
  // Skip the chance/deal phase: mark dealing complete and assign hands directly.
  const hands = Array.from({ length: config.num_players }, () =>
    new Array<number>(config.hand_length).fill(1),
  );
  hands[actingSeat] = expandHand(handCounts, config.hand_length);
  state = { ...state, hands, deal_step: config.total_cards };
  for (const a of sequence) {
    if (!isPlayerNode(state, config)) break;
    const legal = legalActionsMask(state, config);
    if (!legal[a]) break; // defensive: skip an action that became illegal
    state = applyAction(state, config, a);
  }
  return state;
}

function playerLabel(p: number, actingSeat: number): string {
  return p === actingSeat ? `P${p} (acting / AI)` : `P${p}`;
}

export default function PolicyExplorer({ config, serverUrl, serverLabel, onBack }: Props) {
  const { num_players, hand_length, num_digits, max_bids } = config;

  const [actingSeat, setActingSeat] = useState(0);
  // handCounts[d] = how many of digit (d+1) the acting player holds.
  const [handCounts, setHandCounts] = useState<number[]>(() => {
    const c = new Array<number>(num_digits).fill(0);
    c[0] = hand_length; // default: all of digit 1
    return c;
  });
  const [sequence, setSequence] = useState<number[]>([]);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [querying, setQuerying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showJson, setShowJson] = useState(false);

  const handTotal = handCounts.reduce((a, b) => a + b, 0);
  const handValid = handTotal === hand_length;

  // Replay the sequence to the current constructed state.
  const state = useMemo(
    () => buildState(config, sequence, actingSeat, handCounts),
    [config, sequence, actingSeat, handCounts],
  );

  const legal = useMemo(() => legalActionsMask(state, config), [state, config]);
  const terminal = isTerminal(state);
  const atDecision = isPlayerNode(state, config);
  const toMove = state.current_player;
  const isActingTurn = atDecision && toMove === actingSeat;

  // ------------------------------------------------------------------
  // Hand editing
  // ------------------------------------------------------------------
  const bump = useCallback((digitIdx: number, delta: number) => {
    setHandCounts(prev => {
      const next = [...prev];
      next[digitIdx] = Math.max(0, next[digitIdx] + delta);
      return next;
    });
  }, []);

  // ------------------------------------------------------------------
  // Move-sequence editing
  // ------------------------------------------------------------------
  const appendMove = useCallback((action: number) => {
    setSequence(prev => [...prev, action]);
  }, []);
  const undoMove = useCallback(() => setSequence(prev => prev.slice(0, -1)), []);
  const clearMoves = useCallback(() => setSequence([]), []);

  // ------------------------------------------------------------------
  // Auto-query on any change once the state is a queryable decision node.
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!handValid || !isActingTurn) {
      setResult(null);
      return;
    }
    let cancelled = false;
    setQuerying(true);
    setError(null);
    // Greedy=true so the returned `action` marks the argmax on the heatmap; we
    // display the full policy regardless.
    chooseServerAction(serverUrl, state, config, { greedy: true })
      .then(move => {
        if (cancelled) return;
        setResult({
          policy: move.policy,
          value: move.value,
          legal: move.legal,
          action: move.action,
        });
      })
      .catch(e => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setResult(null);
      })
      .finally(() => {
        if (!cancelled) setQuerying(false);
      });
    return () => {
      cancelled = true;
    };
  }, [serverUrl, state, config, handValid, isActingTurn]);

  // ------------------------------------------------------------------
  // Move-history labels (replay to derive whose move each was)
  // ------------------------------------------------------------------
  const historyRows = useMemo(() => {
    const rows: { player: number; label: string }[] = [];
    let s = newInitialState(config);
    const hands = Array.from({ length: num_players }, () =>
      new Array<number>(hand_length).fill(1),
    );
    hands[actingSeat] = expandHand(handCounts, hand_length);
    s = { ...s, hands, deal_step: config.total_cards };
    for (const a of sequence) {
      if (!isPlayerNode(s, config)) break;
      const p = s.current_player;
      let label: string;
      if (a === CHALLENGE_ACTION) {
        label = 'CHALLENGE';
      } else {
        const d = decodeBid(a - 1, config);
        label = `bid ${d.count} × ${d.number}`;
      }
      rows.push({ player: p, label });
      s = applyAction(s, config, a);
    }
    return rows;
  }, [config, sequence, actingSeat, handCounts, num_players, hand_length]);

  // ------------------------------------------------------------------
  // Legal-bid grid (rows = count, cols = digit); enabled iff legal.
  // ------------------------------------------------------------------
  const maxCount = hand_length * num_players;
  const challengeLegal = legal[CHALLENGE_ACTION];

  // Current bid (for the "must exceed" hint).
  const curBid =
    state.current_bid_action >= 0 ? decodeBid(state.current_bid_action - 1, config) : null;

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 p-4">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Policy Explorer</h1>
            <p className="text-gray-400 text-sm">
              Construct a state and query the transformer's action distribution.{' '}
              {serverLabel && <span className="text-gray-500">({serverLabel})</span>}
            </p>
          </div>
          <button
            onClick={onBack}
            className="px-3 py-1.5 rounded-lg bg-gray-700 text-gray-200 hover:bg-gray-600 text-sm"
          >
            ← Back
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* ---------------- LEFT: state builder ---------------- */}
          <div className="space-y-4">
            {/* Acting seat */}
            <div className="bg-gray-800 rounded-xl p-4">
              <label className="block text-sm text-gray-400 mb-2">
                Acting seat (whose policy to query)
              </label>
              <div className="flex flex-wrap gap-2">
                {Array.from({ length: num_players }, (_, i) => (
                  <button
                    key={i}
                    onClick={() => setActingSeat(i)}
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
                      actingSeat === i
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    P{i}
                    {i === 0 ? ' (opens)' : ''}
                  </button>
                ))}
              </div>
              <p className="text-gray-500 text-xs mt-2">
                P0 opens each round. Query fires when it is this seat's turn to move.
              </p>
            </div>

            {/* Hand editor */}
            <div className="bg-gray-800 rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm text-gray-400">
                  Acting player's hand (P{actingSeat}) — count of each digit
                </label>
                <span
                  className={`text-sm font-mono ${
                    handValid ? 'text-green-400' : 'text-red-400'
                  }`}
                >
                  {handTotal} / {hand_length}
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {handCounts.map((c, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between bg-gray-700/60 rounded-lg px-2 py-1.5"
                  >
                    <span className="text-gray-300 font-mono w-6 text-center">{i + 1}</span>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => bump(i, -1)}
                        className="w-6 h-6 rounded bg-gray-600 hover:bg-gray-500 text-white leading-none"
                      >
                        −
                      </button>
                      <span className="w-6 text-center font-mono text-white">{c}</span>
                      <button
                        onClick={() => bump(i, +1)}
                        disabled={handTotal >= hand_length}
                        className="w-6 h-6 rounded bg-gray-600 hover:bg-gray-500 text-white leading-none disabled:opacity-40"
                      >
                        +
                      </button>
                    </div>
                  </div>
                ))}
              </div>
              {!handValid && (
                <p className="text-red-400 text-xs mt-2">
                  Hand must sum to exactly {hand_length} cards.
                </p>
              )}
              <p className="text-gray-500 text-xs mt-2">
                The opponent's hand is hidden/arbitrary — the transformer only sees its own hand.
              </p>
            </div>

            {/* Move builder */}
            <div className="bg-gray-800 rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm text-gray-400">Move sequence</label>
                <div className="flex gap-2">
                  <button
                    onClick={undoMove}
                    disabled={sequence.length === 0}
                    className="px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-xs disabled:opacity-40"
                  >
                    Undo
                  </button>
                  <button
                    onClick={clearMoves}
                    disabled={sequence.length === 0}
                    className="px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-xs disabled:opacity-40"
                  >
                    Clear
                  </button>
                </div>
              </div>

              {/* Applied history */}
              {historyRows.length === 0 ? (
                <p className="text-gray-500 text-xs mb-3">
                  No moves — opening decision (P0 to open).
                </p>
              ) : (
                <ol className="text-sm mb-3 space-y-0.5">
                  {historyRows.map((r, i) => (
                    <li key={i} className="flex gap-2">
                      <span className="text-gray-500 w-6 text-right">{i + 1}.</span>
                      <span className="text-gray-400 w-24">{playerLabel(r.player, actingSeat)}</span>
                      <span className="text-white font-mono">{r.label}</span>
                    </li>
                  ))}
                </ol>
              )}

              {/* Whose turn / status */}
              <div className="text-sm mb-3">
                {terminal ? (
                  <span className="text-yellow-400">
                    Sequence is terminal (a challenge resolved). Undo to reach a decision node.
                  </span>
                ) : (
                  <span className="text-gray-300">
                    To move:{' '}
                    <span
                      className={
                        toMove === actingSeat ? 'text-blue-400 font-medium' : 'text-orange-400 font-medium'
                      }
                    >
                      {playerLabel(toMove, actingSeat)}
                    </span>
                    {curBid && (
                      <span className="text-gray-500">
                        {' '}· current bid {curBid.count} × {curBid.number}
                        {state.is_rebid ? ' (rebid)' : ''}
                      </span>
                    )}
                  </span>
                )}
              </div>

              {/* Move picker */}
              {atDecision && (
                <div>
                  <div className="text-xs text-gray-500 mb-1">
                    Append a move (for {playerLabel(toMove, actingSeat)}):
                  </div>
                  {challengeLegal && (
                    <button
                      onClick={() => appendMove(CHALLENGE_ACTION)}
                      className="mb-2 px-3 py-1.5 rounded-lg bg-red-700 hover:bg-red-600 text-white text-sm font-medium"
                    >
                      CHALLENGE
                    </button>
                  )}
                  {/* digit header */}
                  <div className="flex gap-px mb-0.5" style={{ marginLeft: '1.75rem' }}>
                    {Array.from({ length: num_digits }, (_, i) => (
                      <div key={i} className="text-center text-gray-500 text-xs" style={{ width: 28 }}>
                        {i + 1}
                      </div>
                    ))}
                  </div>
                  {Array.from({ length: maxCount }, (_, ci) => {
                    // Only render rows that contain at least one legal bid.
                    const rowHasLegal = Array.from({ length: num_digits }, (_, di) => {
                      const action = ci * num_digits + di + 1;
                      return action <= max_bids && legal[action];
                    });
                    if (!rowHasLegal.some(Boolean)) return null;
                    return (
                      <div key={ci} className="flex items-center gap-px mb-px">
                        <div className="text-gray-500 text-right mr-1 text-xs" style={{ width: '1.5rem' }}>
                          {ci + 1}×
                        </div>
                        {rowHasLegal.map((isLegal, di) => {
                          const action = ci * num_digits + di + 1;
                          return (
                            <button
                              key={di}
                              disabled={!isLegal}
                              onClick={() => appendMove(action)}
                              title={`bid ${ci + 1} × ${di + 1}`}
                              className={`text-xs rounded-sm ${
                                isLegal
                                  ? 'bg-blue-900/70 hover:bg-blue-600 text-blue-100'
                                  : 'bg-gray-800 text-gray-700 cursor-default'
                              }`}
                              style={{ width: 28, height: 20 }}
                            >
                              {isLegal ? `${ci + 1}·${di + 1}` : ''}
                            </button>
                          );
                        })}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          {/* ---------------- RIGHT: query result ---------------- */}
          <div className="space-y-4">
            <div className="bg-gray-800 rounded-xl p-4">
              <h2 className="text-sm text-gray-400 mb-2">Transformer policy at this state</h2>

              {!handValid ? (
                <p className="text-red-400 text-sm">Fix the hand (must sum to {hand_length}).</p>
              ) : terminal ? (
                <p className="text-yellow-400 text-sm">Terminal state — nothing to query.</p>
              ) : !isActingTurn ? (
                <p className="text-orange-400 text-sm">
                  It is {playerLabel(toMove, actingSeat)}'s turn — add {playerLabel(toMove, actingSeat)}'s
                  move to reach P{actingSeat}'s decision (or switch the acting seat).
                </p>
              ) : querying ? (
                <p className="text-gray-400 text-sm">Querying…</p>
              ) : error ? (
                <p className="text-red-400 text-sm">Error: {error}</p>
              ) : result ? (
                <div>
                  <div className="flex items-center gap-4 mb-1 text-sm">
                    {result.value !== undefined && (
                      <span className="text-gray-300">
                        value: <span className="font-mono text-white">{result.value.toFixed(4)}</span>
                      </span>
                    )}
                    <span className="text-gray-300">
                      argmax:{' '}
                      <span className="font-mono text-white">
                        {result.action === CHALLENGE_ACTION
                          ? 'CHALLENGE'
                          : (() => {
                              const d = decodeBid(result.action - 1, config);
                              return `${d.count} × ${d.number}`;
                            })()}
                      </span>
                    </span>
                    <span className="text-gray-500">
                      legal: {(result.legal ?? legal).filter(Boolean).length} actions
                    </span>
                  </div>
                  <PolicyHeatmap policy={result.policy} takenAction={result.action} config={config} />
                </div>
              ) : (
                <p className="text-gray-500 text-sm">—</p>
              )}
            </div>

            {/* Constructed state JSON (transparency/debugging) */}
            <div className="bg-gray-800 rounded-xl p-4">
              <button
                onClick={() => setShowJson(s => !s)}
                className="text-sm text-gray-400 hover:text-gray-200"
              >
                {showJson ? '▼' : '▶'} Constructed GameState JSON
              </button>
              {showJson && (
                <pre className="mt-2 text-xs text-gray-400 bg-gray-950 rounded-lg p-3 overflow-auto max-h-96">
                  {JSON.stringify(
                    {
                      hands: state.hands,
                      deal_step: state.deal_step,
                      current_player: state.current_player,
                      bid_originator: state.bid_originator,
                      current_bid_action: state.current_bid_action,
                      num_challenges: state.num_challenges,
                      is_rebid: state.is_rebid,
                      winner: state.winner,
                      loser: state.loser,
                    },
                    null,
                    2,
                  )}
                </pre>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
