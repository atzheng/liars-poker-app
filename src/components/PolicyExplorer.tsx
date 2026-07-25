/**
 * PolicyExplorer.tsx — analysis/debugging tool (NOT gameplay).
 *
 * TRAJECTORY INSPECTOR. The user chooses an acting seat + that seat's HAND and a
 * SEQUENCE OF MOVES (built from a fresh deal, or pre-loaded from a live game via
 * the "Inspect in Policy Explorer" button). The Explorer then reconstructs EVERY
 * intermediate GameState along the sequence (reusing src/game.ts via
 * src/trajectory.ts) and queries the trained agent for its action distribution
 * at each decision point where it is the acting seat's turn — not just the
 * final state. Each per-step policy is rendered with PolicyHeatmap.
 *
 * The agent is reached through a `PolicySource` (src/policySource.ts), so the
 * Explorer works against either backend: the Python inference server or an
 * in-browser checkpoint's own forward pass.
 *
 * Editing is reactive: changing the hand, or any move (or adding/removing trailing
 * moves), re-queries every affected step. Re-queries are DEBOUNCED (~300ms) so a
 * burst of edits fires a single round of requests rather than one per keystroke.
 *
 * State construction (see src/trajectory.ts): hands do not affect legality or
 * non-terminal transitions, and the agent only observes hands[current_player], so
 * we build each state with the acting seat's chosen hand injected and placeholder
 * hands for the other seats — the acting seat's queried policy is exact.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { GameConfig } from '../types';
import { CHALLENGE_ACTION, legalActionsMask } from '../game';
import type { PolicySource, PolicyQuery } from '../policySource';
import {
  buildState,
  buildTrajectory,
  handToCounts,
  moveLabel,
  resolveHand,
  sanitizeSequence,
  type TimelineNode,
} from '../trajectory';
import PolicyHeatmap from './PolicyHeatmap';

/** Pre-loaded trajectory captured from a live game (see App.handleInspect). */
export interface ExplorerInit {
  actingSeat: number;
  /** The seat's ACTUAL dealt hand, ordered (legacy checkpoints observe order). */
  hand: number[];        // length hand_length
  sequence: number[];    // bid/challenge action ids, in order
}

interface Props {
  config: GameConfig;
  /** How to evaluate a state — server-backed or in-browser. */
  query: PolicySource;
  /** Short description of the agent being inspected (shown in the header). */
  agentLabel?: string;
  onBack: () => void;
  backLabel?: string;
  /** When set, pre-loads the Explorer with a captured game trajectory. */
  initial?: ExplorerInit;
}

/** Debounce a (referentially-stable) value: returns the last value that has been
 *  unchanged for `delay` ms. React state arrays keep a stable identity between
 *  renders, so this only re-fires when the value actually changes. */
function useDebounced<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return debounced;
}

function playerLabel(p: number, actingSeat: number): string {
  return p === actingSeat ? `P${p} (agent)` : `P${p} (opponent)`;
}

export default function PolicyExplorer({
  config, query, agentLabel, onBack, backLabel, initial,
}: Props) {
  const { num_players, hand_length, num_digits, max_bids } = config;

  const [actingSeat, setActingSeat] = useState(initial?.actingSeat ?? 0);
  // handCounts[d] = how many of digit (d+1) the acting player holds.
  const [handCounts, setHandCounts] = useState<number[]>(() => {
    if (initial?.hand && initial.hand.length === hand_length) {
      return handToCounts(initial.hand, num_digits);
    }
    const c = new Array<number>(num_digits).fill(0);
    c[0] = hand_length; // default: all of digit 1
    return c;
  });
  const [sequence, setSequence] = useState<number[]>(
    initial?.sequence ? [...initial.sequence] : [],
  );
  // Per-step query results, keyed by prefixLen (the # of moves before the node).
  const [results, setResults] = useState<Map<number, PolicyQuery>>(new Map());
  const [querying, setQuerying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [showJson, setShowJson] = useState(false);

  const handTotal = handCounts.reduce((a, b) => a + b, 0);
  const handValid = handTotal === hand_length;

  // States are built from an ORDERED hand: while the counts still match the
  // hand this Explorer was seeded with (the "inspect" flow), keep that exact
  // ordering, so a legacy raw-digit agent sees what it saw in the live game.
  const hand = useMemo(
    () => resolveHand(handCounts, hand_length, initial?.hand),
    [handCounts, hand_length, initial?.hand],
  );

  // ------------------------------------------------------------------
  // Live trajectory (drives structure immediately, no debounce).
  // ------------------------------------------------------------------
  const trajectory = useMemo(
    () => buildTrajectory(config, sequence, actingSeat, hand),
    [config, sequence, actingSeat, hand],
  );
  const finalState = useMemo(
    () => buildState(config, sequence, actingSeat, hand),
    [config, sequence, actingSeat, hand],
  );
  const finalLegal = useMemo(() => legalActionsMask(finalState, config), [finalState, config]);

  // ------------------------------------------------------------------
  // Debounced inputs → drive the agent queries.
  // ------------------------------------------------------------------
  const dSeq = useDebounced(sequence, 300);
  const dHand = useDebounced(hand, 300);
  const dSeat = useDebounced(actingSeat, 300);
  const settled = dSeq === sequence && dHand === hand && dSeat === actingSeat;

  useEffect(() => {
    if (!handValid) {
      setResults(new Map());
      setError(null);
      setQuerying(false);
      return;
    }
    const { nodes } = buildTrajectory(config, dSeq, dSeat, dHand);
    const decisionNodes = nodes.filter(n => n.isActing);
    if (decisionNodes.length === 0) {
      setResults(new Map());
      setQuerying(false);
      return;
    }
    let cancelled = false;
    setQuerying(true);
    setError(null);
    // `action` marks the agent's greedy choice; we render the full policy.
    Promise.all(
      decisionNodes.map(n =>
        query(n.state).then(result => ({ prefixLen: n.prefixLen, result })),
      ),
    )
      .then(entries => {
        if (cancelled) return;
        const map = new Map<number, PolicyQuery>();
        for (const e of entries) map.set(e.prefixLen, e.result);
        setResults(map);
      })
      .catch(e => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setQuerying(false);
      });
    return () => { cancelled = true; };
  }, [query, config, dSeq, dHand, dSeat, handValid]);

  const updating = !settled || querying;

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
  // Move-sequence editing (all keep the sequence engine-valid).
  // ------------------------------------------------------------------
  const appendMove = useCallback((action: number) => {
    setSequence(prev => [...prev, action]);
    setEditingIndex(null);
  }, []);
  const undoMove = useCallback(() => setSequence(prev => prev.slice(0, -1)), []);
  const clearMoves = useCallback(() => { setSequence([]); setEditingIndex(null); }, []);

  // Replace the move at index i, keeping trailing moves that remain legal.
  const replaceMove = useCallback((i: number, action: number) => {
    setSequence(prev => {
      const raw = [...prev.slice(0, i), action, ...prev.slice(i + 1)];
      return sanitizeSequence(config, actingSeat, hand, raw);
    });
    setEditingIndex(null);
  }, [config, actingSeat, hand]);

  // Remove the move at index i, keeping the trailing moves that remain legal.
  const removeMove = useCallback((i: number) => {
    setSequence(prev => {
      const raw = [...prev.slice(0, i), ...prev.slice(i + 1)];
      return sanitizeSequence(config, actingSeat, hand, raw);
    });
    setEditingIndex(null);
  }, [config, actingSeat, hand]);

  // ------------------------------------------------------------------
  // Legal-bid grid (rows = count, cols = digit) for a given state's mask.
  // ------------------------------------------------------------------
  const maxCount = hand_length * num_players;
  const renderBidPicker = (legal: boolean[], onPick: (a: number) => void) => (
    <div>
      {legal[CHALLENGE_ACTION] && (
        <button
          onClick={() => onPick(CHALLENGE_ACTION)}
          className="mb-2 px-3 py-1.5 rounded-lg bg-red-700 hover:bg-red-600 text-white text-sm font-medium"
        >
          CHALLENGE
        </button>
      )}
      <div className="flex gap-px mb-0.5" style={{ marginLeft: '1.75rem' }}>
        {Array.from({ length: num_digits }, (_, i) => (
          <div key={i} className="text-center text-gray-500 text-xs" style={{ width: 28 }}>
            {i + 1}
          </div>
        ))}
      </div>
      {Array.from({ length: maxCount }, (_, ci) => {
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
                  onClick={() => onPick(action)}
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
  );

  // ------------------------------------------------------------------
  // Render one timeline node (agent decision → heatmap; opponent → context row).
  // ------------------------------------------------------------------
  const renderNode = (node: TimelineNode) => {
    const step = node.prefixLen + 1;
    const isLive = node.takenAction === null;
    const takenLbl = node.takenAction !== null ? moveLabel(node.takenAction, config) : null;

    if (!node.isActing) {
      // Opponent move — shown as compact context between the agent's decisions.
      return (
        <div
          key={`n${node.prefixLen}`}
          className="flex items-center gap-2 text-sm text-gray-400 border-l-2 border-gray-700 pl-3 py-1"
        >
          <span className="text-gray-600 w-10 shrink-0">#{step}</span>
          <span className="w-28 shrink-0 text-orange-400/80">{playerLabel(node.toMove, actingSeat)}</span>
          {takenLbl ? (
            <span className="font-mono text-gray-300">{takenLbl}</span>
          ) : (
            <span className="italic text-gray-500">to move (opponent)</span>
          )}
          {node.takenAction !== null && (
            <span className="ml-auto flex gap-1">
              <button
                onClick={() => setEditingIndex(editingIndex === node.prefixLen ? null : node.prefixLen)}
                className="px-1.5 py-0.5 rounded bg-gray-700 hover:bg-gray-600 text-xs"
              >✎</button>
              <button
                onClick={() => removeMove(node.prefixLen)}
                className="px-1.5 py-0.5 rounded bg-gray-700 hover:bg-red-700 text-xs"
              >✕</button>
            </span>
          )}
          {editingIndex === node.prefixLen && (
            <div className="w-full mt-2">
              {renderBidPicker(legalActionsMask(node.state, config), a => replaceMove(node.prefixLen, a))}
            </div>
          )}
        </div>
      );
    }

    // Agent decision node — query result + heatmap.
    const result = results.get(node.prefixLen);
    return (
      <div key={`n${node.prefixLen}`} className="bg-gray-800 rounded-xl p-3 border border-blue-900/50">
        <div className="flex items-center gap-2 mb-1 text-sm flex-wrap">
          <span className="text-gray-500">#{step}</span>
          <span className="text-blue-400 font-medium">{playerLabel(node.toMove, actingSeat)}</span>
          {node.curBid && (
            <span className="text-gray-500 text-xs">
              vs bid {node.curBid.count} × {node.curBid.number}
              {node.state.is_rebid ? ' (rebid)' : ''}
            </span>
          )}
          {isLive
            ? <span className="text-green-400 text-xs">· live decision</span>
            : <span className="text-gray-400 text-xs">· played <span className="font-mono text-yellow-400">{takenLbl}</span></span>}
          {result?.value !== undefined && (
            <span className="text-gray-500 text-xs">· value {result.value.toFixed(3)}</span>
          )}
          {result && (
            <span className="text-gray-500 text-xs">
              · argmax <span className="font-mono text-gray-300">{moveLabel(result.action, config)}</span>
            </span>
          )}
          {node.takenAction !== null && (
            <span className="ml-auto flex gap-1">
              <button
                onClick={() => setEditingIndex(editingIndex === node.prefixLen ? null : node.prefixLen)}
                className="px-1.5 py-0.5 rounded bg-gray-700 hover:bg-gray-600 text-xs"
              >✎ edit</button>
              <button
                onClick={() => removeMove(node.prefixLen)}
                className="px-1.5 py-0.5 rounded bg-gray-700 hover:bg-red-700 text-xs"
              >✕</button>
            </span>
          )}
        </div>

        {editingIndex === node.prefixLen && node.takenAction !== null ? (
          <div className="my-2 p-2 bg-gray-950 rounded-lg">
            <div className="text-xs text-gray-500 mb-1">Replace move #{step} (trailing moves kept where still legal):</div>
            {renderBidPicker(legalActionsMask(node.state, config), a => replaceMove(node.prefixLen, a))}
          </div>
        ) : result ? (
          <div className={updating ? 'opacity-60 transition-opacity' : 'transition-opacity'}>
            {/* Highlight the move actually played (or the argmax for the live node). */}
            <PolicyHeatmap
              policy={result.policy}
              takenAction={node.takenAction ?? result.action}
              config={config}
            />
          </div>
        ) : error ? (
          <p className="text-red-400 text-sm">Error: {error}</p>
        ) : (
          <p className="text-gray-500 text-sm">Querying…</p>
        )}
      </div>
    );
  };

  // Append picker anchored on the final state (adds a trailing move).
  const finalTerminal = trajectory.terminalState !== null;
  const finalIsPlayerNode = !finalTerminal && trajectory.nodes.length > 0
    && trajectory.nodes[trajectory.nodes.length - 1].prefixLen === sequence.length
    && trajectory.nodes[trajectory.nodes.length - 1].takenAction === null;

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Policy Explorer</h1>
            <p className="text-gray-400 text-sm">
              Agent policy at every decision point along a trajectory.{' '}
              {agentLabel && <span className="text-gray-500">({agentLabel})</span>}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {updating && <span className="text-xs text-blue-400 animate-pulse">updating…</span>}
            <button
              onClick={onBack}
              className="px-3 py-1.5 rounded-lg bg-gray-700 text-gray-200 hover:bg-gray-600 text-sm"
            >
              ← {backLabel ?? 'Back'}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* ---------------- LEFT: state builder ---------------- */}
          <div className="space-y-4">
            {/* Acting seat */}
            <div className="bg-gray-800 rounded-xl p-4">
              <label className="block text-sm text-gray-400 mb-2">
                Acting seat (whose policy to inspect)
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
                    P{i}{i === 0 ? ' (opens)' : ''}
                  </button>
                ))}
              </div>
              <p className="text-gray-500 text-xs mt-2">
                Policies are shown at each step where this seat is to move.
              </p>
            </div>

            {/* Hand editor */}
            <div className="bg-gray-800 rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm text-gray-400">
                  Agent's hand (P{actingSeat}) — count of each digit
                </label>
                <span className={`text-sm font-mono ${handValid ? 'text-green-400' : 'text-red-400'}`}>
                  {handTotal} / {hand_length}
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {handCounts.map((c, i) => (
                  <div key={i} className="flex items-center justify-between bg-gray-700/60 rounded-lg px-2 py-1.5">
                    <span className="text-gray-300 font-mono w-6 text-center">{i + 1}</span>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => bump(i, -1)}
                        className="w-6 h-6 rounded bg-gray-600 hover:bg-gray-500 text-white leading-none"
                      >−</button>
                      <span className="w-6 text-center font-mono text-white">{c}</span>
                      <button
                        onClick={() => bump(i, +1)}
                        disabled={handTotal >= hand_length}
                        className="w-6 h-6 rounded bg-gray-600 hover:bg-gray-500 text-white leading-none disabled:opacity-40"
                      >+</button>
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
                Editing the hand re-queries the policy at every step. Opponent hands are
                hidden/arbitrary — the agent only sees its own hand.
              </p>
            </div>

            {/* Sequence-level controls + append picker */}
            <div className="bg-gray-800 rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm text-gray-400">Move sequence ({sequence.length})</label>
                <div className="flex gap-2">
                  <button
                    onClick={undoMove}
                    disabled={sequence.length === 0}
                    className="px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-xs disabled:opacity-40"
                  >Undo last</button>
                  <button
                    onClick={clearMoves}
                    disabled={sequence.length === 0}
                    className="px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-xs disabled:opacity-40"
                  >Clear</button>
                </div>
              </div>

              {finalTerminal ? (
                <p className="text-yellow-400 text-xs">
                  Sequence is terminal (a challenge resolved). Undo / remove a move to reach a
                  decision node.
                </p>
              ) : finalIsPlayerNode ? (
                <div>
                  <div className="text-xs text-gray-500 mb-1">
                    Append a move for {playerLabel(finalState.current_player, actingSeat)}:
                  </div>
                  {renderBidPicker(finalLegal, appendMove)}
                </div>
              ) : (
                <p className="text-gray-500 text-xs">No further move to append.</p>
              )}
            </div>

            {/* Constructed final-state JSON (transparency/debugging) */}
            <div className="bg-gray-800 rounded-xl p-4">
              <button
                onClick={() => setShowJson(s => !s)}
                className="text-sm text-gray-400 hover:text-gray-200"
              >
                {showJson ? '▼' : '▶'} Final GameState JSON
              </button>
              {showJson && (
                <pre className="mt-2 text-xs text-gray-400 bg-gray-950 rounded-lg p-3 overflow-auto max-h-96">
                  {JSON.stringify(
                    {
                      hands: finalState.hands,
                      deal_step: finalState.deal_step,
                      current_player: finalState.current_player,
                      bid_originator: finalState.bid_originator,
                      current_bid_action: finalState.current_bid_action,
                      num_challenges: finalState.num_challenges,
                      is_rebid: finalState.is_rebid,
                      winner: finalState.winner,
                      loser: finalState.loser,
                    }, null, 2,
                  )}
                </pre>
              )}
            </div>
          </div>

          {/* ---------------- RIGHT: per-step trajectory ---------------- */}
          <div className="space-y-2">
            <h2 className="text-sm text-gray-400">
              Trajectory ({trajectory.nodes.filter(n => n.isActing).length} agent decision
              {trajectory.nodes.filter(n => n.isActing).length === 1 ? '' : 's'})
            </h2>
            {!handValid ? (
              <p className="text-red-400 text-sm">Fix the hand (must sum to {hand_length}).</p>
            ) : (
              <div className="space-y-2">
                {trajectory.nodes.map(renderNode)}
                {trajectory.terminalState && (
                  <div className="text-yellow-400 text-sm border-l-2 border-yellow-700 pl-3 py-1">
                    Terminal — the challenge resolved (
                    {trajectory.terminalState.winner >= 0
                      ? `bidder P${trajectory.terminalState.bid_originator} wins`
                      : `bidder P${trajectory.terminalState.bid_originator} loses`}
                    ).
                  </div>
                )}
                {trajectory.nodes.filter(n => n.isActing).length === 0 && (
                  <p className="text-gray-500 text-sm">
                    No decision points for P{actingSeat} yet — append moves, or switch the acting seat.
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
