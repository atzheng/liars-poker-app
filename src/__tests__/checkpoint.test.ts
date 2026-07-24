/**
 * checkpoint.test.ts — the loader must handle BOTH checkpoint generations.
 *
 * 1. Architecture fixtures — tiny checkpoints built by
 *    tests/generate_arch_fixtures.py with the REAL Flax modules, plus the
 *    observations / legal masks / policies the real JAX forward pass produced
 *    for a set of random states. Covers factored_mlp + compact history and
 *    mlp + sparse history, both with the per-digit histogram hand.
 *
 * 2. The legacy preset (public/agents/3x3.msgpack): no architecture metadata,
 *    so it must still load as a flat mlp with the raw-digit hand and sparse
 *    history.
 */

import { readFileSync } from 'fs';
import { join } from 'path';
import { describe, expect, it } from 'vitest';

import { loadCheckpointBytes, observationSize } from '../checkpoint';
import { buildObservation, legalActionsMask } from '../game';
import { networkForward } from '../network';
import { localPolicySource } from '../policySource';
import type { GameState } from '../types';

function readCheckpoint(path: string) {
  const buf = readFileSync(path);
  const bytes = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return loadCheckpointBytes(bytes as ArrayBuffer);
}

// ─── 1. Architecture fixtures ───────────────────────────────────────────────

interface ArchCase {
  state: GameState;
  observer: number;
  obs: number[];
  legal: boolean[];
  policy: number[];
  value: number;
}

interface ArchVariant {
  name: string;
  checkpoint: string;
  network_type: 'mlp' | 'factored_mlp';
  history_encoding: 'sparse' | 'compact';
  hand_encoding: 'digits' | 'histogram';
  max_jump: number | null;
  num_players: number;
  hand_length: number;
  num_digits: number;
  num_actions: number;
  obs_size: number;
  cases: ArchCase[];
}

const TESTS_DIR = join(__dirname, '../../tests');
const archFixtures: { variants: ArchVariant[] } = JSON.parse(
  readFileSync(join(TESTS_DIR, 'arch_fixtures.json'), 'utf-8'),
);

archFixtures.variants.forEach(variant => {
  describe(`checkpoint – ${variant.name} (${variant.network_type}, ${variant.history_encoding})`, () => {
    const data = readCheckpoint(join(TESTS_DIR, variant.checkpoint));

    it('detects the architecture from the checkpoint', () => {
      expect(data.arch.networkType).toBe(variant.network_type);
      expect(data.arch.historyEncoding).toBe(variant.history_encoding);
      expect(data.arch.handEncoding).toBe(variant.hand_encoding);
      expect(data.arch.maxJump).toBe(variant.max_jump);
      expect(data.config.obs_size).toBe(variant.obs_size);
      expect(data.config.num_actions).toBe(variant.num_actions);
      // Whatever the head was, it must consume the last hidden activation and
      // emit one logit per action.
      expect(data.weights.logit.outSize).toBe(variant.num_actions);
      expect(data.weights.hidden[0].inSize).toBe(variant.obs_size);
    });

    variant.cases.forEach((tc, i) => {
      it(`case ${i}: observation, legal mask and policy match JAX`, () => {
        const obs = buildObservation(tc.state, tc.observer, data.config);
        expect(obs.length, 'obs length').toBe(tc.obs.length);
        for (let k = 0; k < tc.obs.length; k++) {
          expect(obs[k], `obs[${k}]`).toBeCloseTo(tc.obs[k], 5);
        }

        expect(legalActionsMask(tc.state, data.config), 'legal').toEqual(tc.legal);

        const pi = networkForward(obs, tc.legal, data.weights);
        for (let a = 0; a < tc.policy.length; a++) {
          expect(pi[a], `policy[${a}]`).toBeCloseTo(tc.policy[a], 5);
        }
      });
    });

    // What the Policy Explorer runs in-browser: same policy, plus the value
    // head and the greedy action.
    it('local policy source matches JAX (policy, value, greedy action)', async () => {
      const query = localPolicySource(data.weights, data.config);
      for (const tc of variant.cases) {
        const q = await query(tc.state);
        expect(q.value, 'value').toBeCloseTo(tc.value, 4);
        expect(q.action, 'greedy action')
          .toBe(tc.policy.indexOf(Math.max(...tc.policy)));
        for (let a = 0; a < tc.policy.length; a++) {
          expect(q.policy[a], `policy[${a}]`).toBeCloseTo(tc.policy[a], 5);
        }
      }
    });
  });
});

// ─── 2. Legacy checkpoint (no architecture metadata) ────────────────────────

describe('checkpoint – legacy 3x3 preset', () => {
  const data = readCheckpoint(join(__dirname, '../../public/agents/3x3.msgpack'));

  it('falls back to the pre-histogram observation layout', () => {
    expect(data.arch.networkType).toBe('mlp');
    expect(data.arch.handEncoding).toBe('digits');
    expect(data.arch.historyEncoding).toBe('sparse');
    expect(data.arch.maxJump).toBeNull();
    expect(data.config.obs_size).toBe(data.weights.hidden[0].inSize);
    expect(data.weights.logit.outSize).toBe(data.config.num_actions);
  });
});

// ─── observationSize ────────────────────────────────────────────────────────

describe('observationSize', () => {
  it('matches observation_size() in liars_poker_jax.py', () => {
    // 2 players × 6 cards × 10 digits → max_bids = 120
    expect(observationSize(2, 6, 10, 'digits', 'sparse')).toBe(2 + 6 + 2 + 480);
    expect(observationSize(2, 6, 10, 'histogram', 'sparse')).toBe(2 + 10 + 2 + 480);
    // compact history = 3*num_digits + num_players + 5
    expect(observationSize(2, 6, 10, 'histogram', 'compact')).toBe(2 + 10 + 2 + 37);
  });
});
