/**
 * trajectory.test.ts — verifies the Policy Explorer's trajectory reconstruction
 * builds engine-valid intermediate states (via the game.ts engine) and picks the
 * correct acting-seat decision points.
 */

import { describe, expect, it } from 'vitest';

import {
  buildInitialState,
  buildState,
  buildTrajectory,
  expandHand,
  handToCounts,
  moveLabel,
  sanitizeSequence,
} from '../trajectory';
import { CHALLENGE_ACTION, encodeBid, isTerminal, legalActionsMask } from '../game';
import type { GameConfig } from '../types';

function makeConfig(num_players = 2, hand_length = 3, num_digits = 3): GameConfig {
  const total_cards = num_players * hand_length;
  const max_bids = hand_length * num_digits * num_players;
  const num_actions = max_bids + 1;
  const obs_size = num_players + hand_length + 2 + 2 * max_bids * num_players;
  return { num_players, hand_length, num_digits, total_cards, max_bids, num_actions, obs_size };
}

describe('hand helpers', () => {
  it('handToCounts inverts expandHand', () => {
    const cfg = makeConfig();
    const counts = [1, 2, 0]; // one of digit1, two of digit2
    const hand = expandHand(counts, cfg.hand_length);
    expect(hand.length).toBe(cfg.hand_length);
    expect(handToCounts(hand, cfg.num_digits)).toEqual(counts);
  });

  it('handToCounts ignores out-of-range/undealt slots', () => {
    expect(handToCounts([1, 1, 3, 0], 3)).toEqual([2, 0, 1]);
  });
});

describe('buildState / buildInitialState', () => {
  it('post-deal initial state injects acting hand and marks dealing done', () => {
    const cfg = makeConfig();
    const s = buildInitialState(cfg, 1, [0, 3, 0]); // P1 holds three 2s
    expect(s.deal_step).toBe(cfg.total_cards);
    expect(s.current_player).toBe(0); // P0 opens
    expect(s.hands[1]).toEqual([2, 2, 2]);
    expect(legalActionsMask(s, cfg)[CHALLENGE_ACTION]).toBe(false); // no bid yet
  });

  it('replays a prefix and matches the engine step-by-step', () => {
    const cfg = makeConfig();
    const bidA = encodeBid(1, 2, cfg) + 1; // 1 × 2
    const bidB = encodeBid(2, 3, cfg) + 1; // 2 × 3
    const seq = [bidA, bidB];
    const s0 = buildState(cfg, seq, 0, [3, 0, 0], 0);
    expect(s0.current_player).toBe(0);
    const s1 = buildState(cfg, seq, 0, [3, 0, 0], 1);
    expect(s1.current_player).toBe(1);
    expect(s1.current_bid_action).toBe(bidA);
    const s2 = buildState(cfg, seq, 0, [3, 0, 0]);
    expect(s2.current_player).toBe(0);
    expect(s2.current_bid_action).toBe(bidB);
  });
});

describe('buildTrajectory', () => {
  it('opening decision only when the sequence is empty', () => {
    const cfg = makeConfig();
    const t = buildTrajectory(cfg, [], 0, [3, 0, 0]);
    expect(t.terminalState).toBeNull();
    const decisions = t.nodes.filter(n => n.isActing);
    expect(decisions.length).toBe(1);
    expect(decisions[0].prefixLen).toBe(0);
    expect(decisions[0].takenAction).toBeNull(); // live decision, nothing played
  });

  it('marks acting-seat nodes and records the move actually taken', () => {
    const cfg = makeConfig();
    const bidA = encodeBid(1, 1, cfg) + 1;
    const bidB = encodeBid(1, 2, cfg) + 1;
    // P0 opens (bidA), P1 raises (bidB); acting seat = P1.
    const t = buildTrajectory(cfg, [bidA, bidB], 1, [0, 3, 0]);
    expect(t.nodes.length).toBe(3); // prefixes 0,1,2
    expect(t.nodes[0].toMove).toBe(0);
    expect(t.nodes[0].isActing).toBe(false);      // opponent (P0) opening
    expect(t.nodes[0].takenAction).toBe(bidA);
    expect(t.nodes[1].toMove).toBe(1);
    expect(t.nodes[1].isActing).toBe(true);        // P1's decision, played bidB
    expect(t.nodes[1].takenAction).toBe(bidB);
    expect(t.nodes[2].toMove).toBe(0);             // back to P0 (live)
    expect(t.nodes[2].takenAction).toBeNull();
  });

  it('stops at a terminal challenge and reports the outcome', () => {
    const cfg = makeConfig();
    const bidA = encodeBid(6, 1, cfg) + 1; // 6 × 1 — big bid, likely to fail
    // P0 opens huge; P1 challenges; P0 declines to rebid (challenges) → count
    // resolves the game terminal in a 2-player game.
    const t = buildTrajectory(cfg, [bidA, CHALLENGE_ACTION, CHALLENGE_ACTION], 1, [0, 3, 0]);
    expect(t.terminalState).not.toBeNull();
    expect(isTerminal(t.terminalState!)).toBe(true);
    // No node emitted past the terminal.
    expect(t.nodes.every(n => !isTerminal(n.state))).toBe(true);
  });
});

describe('sanitizeSequence', () => {
  it('keeps a legal sequence intact', () => {
    const cfg = makeConfig();
    const bidA = encodeBid(1, 1, cfg) + 1;
    const bidB = encodeBid(2, 1, cfg) + 1;
    const raw = [bidA, bidB];
    expect(sanitizeSequence(cfg, 0, [3, 0, 0], raw)).toEqual(raw);
  });

  it('drops trailing moves that become illegal', () => {
    const cfg = makeConfig();
    const low = encodeBid(1, 1, cfg) + 1;
    const high = encodeBid(3, 3, cfg) + 1;
    // After a high bid, a lower bid is illegal (bids strictly increase) → dropped.
    const kept = sanitizeSequence(cfg, 0, [3, 0, 0], [high, low]);
    expect(kept).toEqual([high]);
  });
});

describe('moveLabel', () => {
  it('labels challenge and bids', () => {
    const cfg = makeConfig();
    expect(moveLabel(CHALLENGE_ACTION, cfg)).toBe('CHALLENGE');
    const bid = encodeBid(2, 3, cfg) + 1;
    expect(moveLabel(bid, cfg)).toBe('2 × 3');
  });
});
