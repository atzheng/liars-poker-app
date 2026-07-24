/**
 * network.ts — Pure TypeScript MLP forward pass.
 *
 * Mirrors RNaDNetwork in rnad.py:
 *   for size in policy_network_layers:
 *     x = relu(Dense(size)(x))
 *   logit = Dense(num_actions)(x)
 *   pi = _legal_policy(logit, legal)
 *
 * FactoredMLPNetwork (the current architecture) replaces the flat logit head
 * with a query/key dot product:
 *   q     = Dense(d)(x)                       # per-state query
 *   key   = Dense(d)(relu(Dense(d)(phi)))     # per-action keys, phi static
 *   logit = q @ key.T / sqrt(d)
 * Because `phi` (and therefore `key`) depends only on the game config, the
 * whole head is affine in x and folds into ONE equivalent Dense layer at load
 * time (foldFactoredHead below). Inference is then byte-for-byte the same code
 * path as the flat head.
 *
 * And _legal_policy from rnad.py:
 *   l_min = logits.min()
 *   logits = where(legal, logits, l_min)
 *   logits -= logits.max()
 *   logits *= legal
 *   exp_logits = where(legal, exp(logits), 0)
 *   return exp_logits / sum(exp_logits)
 */

import type { DenseLayer, GameConfig, NetworkWeights } from './types';

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

function matmulAdd(x: Float32Array, layer: DenseLayer): Float32Array {
  // y[j] = bias[j] + sum_i(x[i] * kernel[i * outSize + j])
  const { kernel, bias, outSize } = layer;
  const y = new Float32Array(outSize);
  for (let j = 0; j < outSize; j++) {
    let s = bias[j];
    for (let i = 0; i < x.length; i++) {
      s += x[i] * kernel[i * outSize + j];
    }
    y[j] = s;
  }
  return y;
}

function relu(x: Float32Array): Float32Array {
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) y[i] = x[i] > 0 ? x[i] : 0;
  return y;
}

// ---------------------------------------------------------------------------
// Factored (query/key) policy head
// ---------------------------------------------------------------------------

/**
 * Static per-action feature table `phi` of shape [num_actions, 2 + num_digits],
 * raveled row-major. Mirrors FactoredMLPNetwork._build_phi in rnad.py:
 *   col 0       : is_challenge (1 for action 0, else 0)
 *   col 1       : count / total_cards          (0 for challenge)
 *   cols 2..2+D : one-hot of (number - 1)      (all zeros for challenge)
 *
 * Note this is the 'linear' count encoding, the only one the inference server
 * (serve_agent_mlp.py) builds — checkpoints do not record which count encoding
 * they were trained with.
 */
export function buildActionFeatures(config: GameConfig): {
  phi: Float32Array;
  dPhi: number;
} {
  const { num_actions, num_digits: D, total_cards } = config;
  const dPhi = 2 + D;
  const phi = new Float32Array(num_actions * dPhi);
  phi[0] = 1; // action 0 == challenge
  for (let a = 1; a < num_actions; a++) {
    const bidId = a - 1;
    const number = (bidId % D) + 1;
    const count = Math.floor(bidId / D) + 1;
    phi[a * dPhi + 1] = count / total_cards;
    phi[a * dPhi + 2 + (number - 1)] = 1;
  }
  return { phi, dPhi };
}

/**
 * Collapse a factored head into the equivalent flat logit Dense layer.
 *
 * With  q = Wq·x + bq  and static keys K [num_actions, d],
 *   logit = q·Kᵀ / sqrt(d) = x·(Wq·Kᵀ/sqrt(d)) + bq·Kᵀ/sqrt(d)
 * so the folded layer is  kernel = Wq·Kᵀ/sqrt(d), bias = bq·Kᵀ/sqrt(d).
 *
 * `query` maps the last hidden activation to the d-dim query; `keyHidden` and
 * `keyOut` are the two Dense layers applied to `phi` (relu in between).
 */
export function foldFactoredHead(
  query: DenseLayer,
  keyHidden: DenseLayer,
  keyOut: DenseLayer,
  config: GameConfig,
): DenseLayer {
  const { phi, dPhi } = buildActionFeatures(config);
  const numActions = config.num_actions;
  const d = keyOut.outSize;

  if (keyHidden.inSize !== dPhi) {
    throw new Error(
      `factored head expects ${dPhi} action features (2 + num_digits), ` +
      `but its key layer takes ${keyHidden.inSize} — checkpoint/game mismatch`,
    );
  }
  if (query.outSize !== d) {
    throw new Error(
      `factored head query dim ${query.outSize} != key dim ${d}`,
    );
  }

  // keys[a] = keyOut(relu(keyHidden(phi[a])))  → [num_actions, d]
  const keys = new Float32Array(numActions * d);
  for (let a = 0; a < numActions; a++) {
    const row = phi.subarray(a * dPhi, (a + 1) * dPhi);
    const hidden = relu(matmulAdd(row, keyHidden));
    keys.set(matmulAdd(hidden, keyOut), a * d);
  }

  // kernel[i, a] = sum_j Wq[i, j] * keys[a, j] / sqrt(d)
  const scale = 1 / Math.sqrt(d);
  const kernel = new Float32Array(query.inSize * numActions);
  for (let i = 0; i < query.inSize; i++) {
    for (let a = 0; a < numActions; a++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += query.kernel[i * d + j] * keys[a * d + j];
      kernel[i * numActions + a] = s * scale;
    }
  }
  const bias = new Float32Array(numActions);
  for (let a = 0; a < numActions; a++) {
    let s = 0;
    for (let j = 0; j < d; j++) s += query.bias[j] * keys[a * d + j];
    bias[a] = s * scale;
  }

  return { kernel, bias, inSize: query.inSize, outSize: numActions };
}

// ---------------------------------------------------------------------------
// Legal policy (masked softmax)
// ---------------------------------------------------------------------------

export function legalPolicy(logits: Float32Array, legal: boolean[]): Float32Array {
  const n = logits.length;
  const masked = new Float32Array(n);
  
  // 1. Apply -Infinity mask and find Max of LEGAL moves only
  let maxLegal = -Infinity;
  for (let i = 0; i < n; i++) {
    if (legal[i]) {
      if (logits[i] > maxLegal) maxLegal = logits[i];
    }
  }

  // If no legal moves exist, you might want to return a uniform distribution 
  // or handle the error, but for now we assume at least one legal move.
  
  // 2. Exp and Sum (using maxLegal for stability)
  let sum = 0;
  for (let i = 0; i < n; i++) {
    if (legal[i]) {
      // exp(logit - max) is numerically stable
      masked[i] = Math.exp(logits[i] - maxLegal);
      sum += masked[i];
    } else {
      masked[i] = 0; // These will have 0 probability
    }
  }

  // 3. Normalize
  for (let i = 0; i < n; i++) {
    masked[i] /= sum;
  }

  return masked;
}

// ---------------------------------------------------------------------------
// Full forward pass
// ---------------------------------------------------------------------------

/** Run the relu-MLP torso, returning the last hidden activation. */
function trunk(obs: Float32Array, weights: NetworkWeights): Float32Array {
  const inSize = weights.hidden[0]?.inSize ?? weights.logit.inSize;
  if (obs.length !== inSize) {
    throw new Error(
      `observation length ${obs.length} does not match the network's input ` +
      `size ${inSize} — the game config does not match this checkpoint`,
    );
  }
  let x = obs;
  for (const layer of weights.hidden) {
    x = relu(matmulAdd(x, layer));
  }
  return x;
}

/** Policy head: masked softmax over the (optionally re-tempered) logits. */
function policyFromHidden(
  x: Float32Array,
  legal: boolean[],
  weights: NetworkWeights,
  temperature: number,
): Float32Array {
  const logits = matmulAdd(x, weights.logit);
  if (temperature !== 1) {
    for (let i = 0; i < logits.length; i++) logits[i] /= temperature;
  }
  return legalPolicy(logits, legal);
}

/** Returns the policy distribution over all actions (legal actions only).
 *  temperature > 1 flattens the distribution; < 1 sharpens it toward greedy. */
export function networkForward(
  obs: Float32Array,
  legal: boolean[],
  weights: NetworkWeights,
  temperature = 1,
): Float32Array {
  return policyFromHidden(trunk(obs, weights), legal, weights, temperature);
}

/**
 * Policy plus the value head's estimate for the observing player — the same
 * pair the inference server returns from /move. `value` is undefined when the
 * checkpoint carries no value head (nothing outside analysis needs it).
 */
export function networkForwardWithValue(
  obs: Float32Array,
  legal: boolean[],
  weights: NetworkWeights,
  temperature = 1,
): { policy: Float32Array; value?: number } {
  const x = trunk(obs, weights);
  const policy = policyFromHidden(x, legal, weights, temperature);
  const value = weights.value ? matmulAdd(x, weights.value)[0] : undefined;
  return { policy, value };
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/** Sample an action from the policy distribution. */
export function sampleAction(policy: Float32Array): number {
  const r = Math.random();
  let cumsum = 0;
  for (let i = 0; i < policy.length; i++) {
    cumsum += policy[i];
    if (r < cumsum) return i;
  }
  return policy.length - 1;
}
