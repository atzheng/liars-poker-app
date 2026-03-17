/**
 * checkpoint.ts — Load a Flax/JAX checkpoint from .msgpack or .json.
 *
 * Flax msgpack format (flax.serialization.to_bytes):
 *   - Arrays are stored as ExtType(1, inner_bytes) where:
 *     inner_bytes = msgpack([shape, dtype_str, raw_bytes])
 *   - Numpy scalars are ExtType(3, inner_bytes) with same inner format.
 *   - All other values (dicts, lists, ints, strings) are plain msgpack.
 *
 * The top-level state dict contains:
 *   params_target.params.Dense_0 .. Dense_N  — network weights
 *   jax_config   — {num_players, hand_length, num_digits}
 *   policy_network_layers — [layer_size, ...]
 */

import { decode, ExtensionCodec } from '@msgpack/msgpack';
import type { GameConfig, DenseLayer, NetworkWeights } from './types';

// ---------------------------------------------------------------------------
// Flax ext-type decoder
// ---------------------------------------------------------------------------

function makeFlaxCodec(): ExtensionCodec {
  const codec = new ExtensionCodec();

  // ExtType 1 = numpy ndarray
  codec.register({
    type: 1,
    encode: () => { throw new Error('encode not supported'); },
    decode(data: Uint8Array) {
      return decodeNdarray(data);
    },
  });

  // ExtType 3 = numpy scalar (stored as 0-d ndarray)
  codec.register({
    type: 3,
    encode: () => { throw new Error('encode not supported'); },
    decode(data: Uint8Array) {
      const arr = decodeNdarray(data);
      if (arr instanceof Float32Array) return arr[0];
      if (arr instanceof Int32Array)   return arr[0];
      if (arr instanceof BigInt64Array) return Number(arr[0]);
      return 0;
    },
  });

  return codec;
}

function decodeNdarray(data: Uint8Array): Float32Array | Int32Array | BigInt64Array | Uint8Array {
  // inner msgpack: [shape: number[], dtype: string, rawBytes: Uint8Array]
  const inner = decode(data) as [number[], string, Uint8Array];
  const [_shape, dtype, rawBytes] = inner;

  // Copy to aligned buffer
  const aligned = rawBytes.buffer.slice(
    rawBytes.byteOffset,
    rawBytes.byteOffset + rawBytes.byteLength,
  );

  if (dtype === 'float32') return new Float32Array(aligned);
  if (dtype === 'int32')   return new Int32Array(aligned);
  if (dtype === 'int64')   return new BigInt64Array(aligned);
  if (dtype === 'uint32')  return new Int32Array(aligned);
  return new Uint8Array(aligned);
}

// ---------------------------------------------------------------------------
// Weight extraction helpers
// ---------------------------------------------------------------------------

function extractDenseLayer(layer: Record<string, unknown>): DenseLayer {
  const kernel = layer['kernel'] as Float32Array;
  const bias   = layer['bias']   as Float32Array;
  const outSize = bias.length;
  const inSize  = kernel.length / outSize;
  return { kernel, bias, inSize, outSize };
}

function buildConfig(
  jaxConfig: { num_players: number; hand_length: number; num_digits: number },
): GameConfig {
  return buildGameConfig(jaxConfig.num_players, jaxConfig.hand_length, jaxConfig.num_digits);
}

export function buildGameConfig(num_players: number, hand_length: number, num_digits: number): GameConfig {
  const total_cards = num_players * hand_length;
  const max_bids    = hand_length * num_digits * num_players;
  const num_actions = max_bids + 1;
  // obs_size = num_players + hand_length + 2 + 2*max_bids*num_players
  const obs_size    = num_players + hand_length + 2 + 2 * max_bids * num_players;
  return { num_players, hand_length, num_digits, total_cards, max_bids, num_actions, obs_size };
}

function extractWeights(
  state: Record<string, unknown>,
  numHiddenLayers: number,
): NetworkWeights {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const paramsTarget = state['params_target'] as any;
  const params = paramsTarget['params'] as Record<string, unknown>;

  const hidden: DenseLayer[] = [];
  for (let i = 0; i < numHiddenLayers; i++) {
    hidden.push(extractDenseLayer(params[`Dense_${i}`] as Record<string, unknown>));
  }
  const logit = extractDenseLayer(params[`Dense_${numHiddenLayers}`] as Record<string, unknown>);

  return { hidden, logit, numLayers: numHiddenLayers };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface CheckpointData {
  config: GameConfig;
  weights: NetworkWeights;
}

/**
 * Flax serializes Python lists/tuples as dicts with string-int keys,
 * e.g. [256, 256, 256] → {'0': 256, '1': 256, '2': 256}.
 * This normalizes either form back to a number[].
 */
function normalizeIntList(value: unknown): number[] {
  if (Array.isArray(value)) return value as number[];
  if (value && typeof value === 'object') {
    const obj = value as Record<string, number>;
    return Object.keys(obj)
      .map(Number)
      .sort((a, b) => a - b)
      .map(k => obj[String(k)]);
  }
  return [];
}

/** Load checkpoint from an ArrayBuffer (msgpack) or string (JSON). */
export function loadCheckpointBytes(buffer: ArrayBuffer): CheckpointData {
  const codec = makeFlaxCodec();
  const state = decode(new Uint8Array(buffer), { extensionCodec: codec }) as Record<string, unknown>;

  const jaxCfg = state['jax_config'] as { num_players: number; hand_length: number; num_digits: number };
  const layers = normalizeIntList(state['policy_network_layers']);

  return {
    config: buildConfig(jaxCfg),
    weights: extractWeights(state, layers.length),
  };
}

export function loadCheckpointJson(json: string): CheckpointData {
  const raw = JSON.parse(json) as {
    jax_config: { num_players: number; hand_length: number; num_digits: number };
    policy_network_layers: number[];
    params_target: Record<string, Record<string, number[][]>>;
  };

  const numHidden = raw.policy_network_layers.length;
  const params = raw.params_target;

  const hidden: DenseLayer[] = [];
  for (let i = 0; i < numHidden; i++) {
    const layer = params[`Dense_${i}`];
    // kernel is [[...], ...] in JSON — flatten to 1D
    const kernel = Float32Array.from((layer['kernel'] as unknown as number[][]).flatMap(r => r));
    const bias   = Float32Array.from(layer['bias'] as unknown as number[]);
    hidden.push({ kernel, bias, inSize: kernel.length / bias.length, outSize: bias.length });
  }

  const logitRaw = params[`Dense_${numHidden}`];
  const logitKernel = Float32Array.from((logitRaw['kernel'] as unknown as number[][]).flatMap(r => r));
  const logitBias   = Float32Array.from(logitRaw['bias'] as unknown as number[]);
  const logit: DenseLayer = {
    kernel: logitKernel,
    bias: logitBias,
    inSize: logitKernel.length / logitBias.length,
    outSize: logitBias.length,
  };

  return {
    config: buildConfig(raw.jax_config),
    weights: { hidden, logit, numLayers: numHidden },
  };
}
