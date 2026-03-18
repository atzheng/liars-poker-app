/**
 * network.ts — Pure TypeScript MLP forward pass.
 *
 * Mirrors RNaDNetwork in rnad.py:
 *   for size in policy_network_layers:
 *     x = relu(Dense(size)(x))
 *   logit = Dense(num_actions)(x)
 *   pi = _legal_policy(logit, legal)
 *
 * And _legal_policy from rnad.py:
 *   l_min = logits.min()
 *   logits = where(legal, logits, l_min)
 *   logits -= logits.max()
 *   logits *= legal
 *   exp_logits = where(legal, exp(logits), 0)
 *   return exp_logits / sum(exp_logits)
 */

import type { DenseLayer, NetworkWeights } from './types';

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
// Legal policy (masked softmax)
// ---------------------------------------------------------------------------

export function legalPolicy(logits: Float32Array, legal: boolean[]): Float32Array {
  const n = logits.length;

  // l_min over all logits
  let lMin = logits[0];
  for (let i = 1; i < n; i++) if (logits[i] < lMin) lMin = logits[i];

  // Replace illegal with l_min
  const masked = new Float32Array(n);
  for (let i = 0; i < n; i++) masked[i] = legal[i] ? logits[i] : lMin;

  // Subtract max for numerical stability
  let lMax = masked[0];
  for (let i = 1; i < n; i++) if (masked[i] > lMax) lMax = masked[i];
  for (let i = 0; i < n; i++) masked[i] -= lMax;

  // Multiply by legal mask (illegal → 0 before exp)
  for (let i = 0; i < n; i++) if (!legal[i]) masked[i] = 0;

  // Exp and sum
  const expLogits = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    expLogits[i] = legal[i] ? Math.exp(masked[i]) : 0;
    sum += expLogits[i];
  }

  // Normalize
  for (let i = 0; i < n; i++) expLogits[i] /= sum;
  return expLogits;
}

// ---------------------------------------------------------------------------
// Full forward pass
// ---------------------------------------------------------------------------

/** Returns the policy distribution over all actions (legal actions only).
 *  temperature > 1 flattens the distribution; < 1 sharpens it toward greedy. */
export function networkForward(
  obs: Float32Array,
  legal: boolean[],
  weights: NetworkWeights,
  temperature = 1,
): Float32Array {
  let x = obs;
  for (const layer of weights.hidden) {
    x = relu(matmulAdd(x, layer));
  }
  const logits = matmulAdd(x, weights.logit);
  if (temperature !== 1) {
    for (let i = 0; i < logits.length; i++) logits[i] /= temperature;
  }
  return legalPolicy(logits, legal);
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
