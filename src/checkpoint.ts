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
 *   jax_config   — {num_players, hand_length, num_digits, max_jump?}
 *   policy_network_layers — [layer_size, ...]
 *   network_type          — 'mlp' | 'factored_mlp' | 'transformer'   (newer)
 *   history_encoding      — 'sparse' | 'compact'                     (newer)
 *
 * Two generations of checkpoints are supported (train_jax.py writes both):
 *
 *   legacy  — no network_type/history_encoding metadata. Flat Dense logit head,
 *             raw-digit hand, sparse history.
 *   current — network_type 'mlp' or 'factored_mlp', history_encoding 'sparse'
 *             or 'compact', per-digit histogram hand, optional max_jump.
 *
 * The hand encoding is NOT recorded in the checkpoint, so it is inferred from
 * the first layer's input width (see inferHandEncoding).
 */

import { decode, ExtensionCodec } from '@msgpack/msgpack';
import { foldFactoredHead } from './network';
import type {
  GameConfig, DenseLayer, HandEncoding, HistoryEncoding, NetworkType,
  NetworkWeights,
} from './types';

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

function extractDenseLayer(
  layer: Record<string, unknown> | undefined,
  name: string,
): DenseLayer {
  const kernel = layer?.['kernel'] as Float32Array | undefined;
  const bias   = layer?.['bias']   as Float32Array | undefined;
  if (!kernel || !bias) {
    throw new Error(`checkpoint is missing the ${name} layer (kernel/bias)`);
  }
  const outSize = bias.length;
  const inSize  = kernel.length / outSize;
  return { kernel, bias, inSize, outSize };
}

/** Optional game/observation settings that come from checkpoint metadata. */
export interface GameConfigOptions {
  maxJump?: number | null;
  firstBidBaseCount?: number | null;
  maxBidCount?: number | null;
  /** Defaults to the legacy 'digits' layout. */
  handEncoding?: HandEncoding;
  /** Defaults to the legacy 'sparse' layout. */
  historyEncoding?: HistoryEncoding;
}

export function buildGameConfig(
  num_players: number,
  hand_length: number,
  num_digits: number,
  opts: GameConfigOptions = {},
): GameConfig {
  const handEncoding = opts.handEncoding ?? 'digits';
  const historyEncoding = opts.historyEncoding ?? 'sparse';
  const total_cards = num_players * hand_length;
  const max_bids    = hand_length * num_digits * num_players;
  const num_actions = max_bids + 1;
  const obs_size    = observationSize(
    num_players, hand_length, num_digits, handEncoding, historyEncoding,
  );
  return {
    num_players, hand_length, num_digits,
    maxJump: opts.maxJump ?? null,
    firstBidBaseCount: opts.firstBidBaseCount ?? null,
    maxBidCount: opts.maxBidCount ?? null,
    handEncoding, historyEncoding,
    total_cards, max_bids, num_actions, obs_size,
  };
}

/**
 * Length of the flat observation tensor — mirrors observation_size() in
 * liars_poker_jax.py:
 *   num_players
 *   + hand_length (digits) | num_digits (histogram)
 *   + 2                                          (is_rebid, is_terminal)
 *   + 2*max_bids*num_players (sparse) | 3*num_digits + num_players + 5 (compact)
 */
export function observationSize(
  num_players: number,
  hand_length: number,
  num_digits: number,
  handEncoding: HandEncoding,
  historyEncoding: HistoryEncoding,
): number {
  const max_bids = hand_length * num_digits * num_players;
  const hand = handEncoding === 'histogram' ? num_digits : hand_length;
  const history = historyEncoding === 'compact'
    ? 3 * num_digits + num_players + 5
    : 2 * max_bids * num_players;
  return num_players + hand + 2 + history;
}

// ---------------------------------------------------------------------------
// Architecture detection
// ---------------------------------------------------------------------------

/** What the checkpoint metadata (plus shape inference) says it is. */
export interface CheckpointArch {
  networkType: NetworkType;
  handEncoding: HandEncoding;
  historyEncoding: HistoryEncoding;
  maxJump: number | null;
  hiddenLayers: number[];
}

/**
 * Infer the hand encoding from the first layer's input width.
 *
 * The hand encoding switched from raw digits to a per-digit histogram without
 * a metadata flag, so the only reliable signal is the observation width the
 * network was built for. When hand_length == num_digits both encodings have
 * the same width; in that case the presence of the newer metadata fields
 * (network_type / history_encoding) decides.
 */
function inferHandEncoding(
  num_players: number,
  hand_length: number,
  num_digits: number,
  historyEncoding: HistoryEncoding,
  inputSize: number,
  hasModernMetadata: boolean,
): HandEncoding {
  const size = (h: HandEncoding) =>
    observationSize(num_players, hand_length, num_digits, h, historyEncoding);
  const digitsFits = size('digits') === inputSize;
  const histogramFits = size('histogram') === inputSize;

  if (digitsFits && histogramFits) return hasModernMetadata ? 'histogram' : 'digits';
  if (histogramFits) return 'histogram';
  if (digitsFits) return 'digits';

  throw new Error(
    `checkpoint's first layer takes ${inputSize} inputs, but a ` +
    `${num_players}p × ${hand_length} cards × ${num_digits} digits game with ` +
    `${historyEncoding} history produces ${size('digits')} (raw-digit hand) or ` +
    `${size('histogram')} (histogram hand). Unsupported observation layout.`,
  );
}

/** Read `network_type`, rejecting architectures the browser cannot run. */
function readNetworkType(raw: unknown): NetworkType {
  const value = (raw as string) ?? 'mlp';
  if (value === 'mlp' || value === 'factored_mlp') return value;
  throw new Error(
    `network_type '${value}' cannot run in the browser (only 'mlp' and ` +
    `'factored_mlp' are supported) — use the inference server for it`,
  );
}

// ---------------------------------------------------------------------------
// Weight extraction
// ---------------------------------------------------------------------------

/**
 * Build the network weights from the flat `Dense_i` params.
 *
 * Layer order follows the Flax modules' call order in rnad.py:
 *   mlp           Dense_0..L-1 hidden, Dense_L logit, Dense_L+1 value
 *   factored_mlp  Dense_0..L-1 hidden, Dense_L query, Dense_L+1 key-hidden,
 *                 Dense_L+2 key-out, Dense_L+3 value
 * The factored head is folded into an equivalent flat logit layer (see
 * foldFactoredHead). The value head is optional: only the Policy Explorer
 * shows it, so a checkpoint without one still loads.
 */
function extractWeights(
  params: Record<string, unknown>,
  numHiddenLayers: number,
  networkType: NetworkType,
  config: GameConfig,
): NetworkWeights {
  const dense = (i: number): DenseLayer =>
    extractDenseLayer(
      params[`Dense_${i}`] as Record<string, unknown> | undefined,
      `Dense_${i}`,
    );

  const hidden: DenseLayer[] = [];
  for (let i = 0; i < numHiddenLayers; i++) hidden.push(dense(i));

  const logit = networkType === 'factored_mlp'
    ? foldFactoredHead(
        dense(numHiddenLayers),      // query
        dense(numHiddenLayers + 1),  // key: phi → d
        dense(numHiddenLayers + 2),  // key: d → d
        config,
      )
    : dense(numHiddenLayers);

  if (logit.outSize !== config.num_actions) {
    throw new Error(
      `checkpoint policy head has ${logit.outSize} actions, but a ` +
      `${config.num_players}p × ${config.hand_length} cards × ` +
      `${config.num_digits} digits game has ${config.num_actions}`,
    );
  }

  // Value head: the Dense right after the policy head. Scalar output, applied
  // to the same last hidden activation.
  const valueIndex = numHiddenLayers + (networkType === 'factored_mlp' ? 3 : 1);
  const valueRaw = params[`Dense_${valueIndex}`] as Record<string, unknown> | undefined;
  let value: DenseLayer | undefined;
  if (valueRaw) {
    const layer = extractDenseLayer(valueRaw, `Dense_${valueIndex}`);
    if (layer.outSize === 1) value = layer;
  }

  return { hidden, logit, value, numLayers: numHiddenLayers, networkType };
}

/**
 * Shared tail of the msgpack/JSON loaders: read the metadata, work out the
 * observation layout, and build the config + weights.
 */
function buildCheckpoint(
  jaxConfig: Record<string, unknown>,
  layers: number[],
  params: Record<string, unknown>,
  metadata: { network_type?: unknown; history_encoding?: unknown },
): CheckpointData {
  const num_players = Number(jaxConfig['num_players']);
  const hand_length = Number(jaxConfig['hand_length']);
  const num_digits  = Number(jaxConfig['num_digits']);
  const maxJumpRaw  = jaxConfig['max_jump'];
  const maxJump     = maxJumpRaw == null ? null : Number(maxJumpRaw);

  const hasModernMetadata =
    metadata.network_type != null || metadata.history_encoding != null;
  const networkType = readNetworkType(metadata.network_type);
  const historyEncoding =
    (metadata.history_encoding as HistoryEncoding) ?? 'sparse';
  if (historyEncoding !== 'sparse' && historyEncoding !== 'compact') {
    throw new Error(`unknown history_encoding '${historyEncoding}'`);
  }

  const firstLayer = extractDenseLayer(
    params['Dense_0'] as Record<string, unknown> | undefined, 'Dense_0',
  );
  const handEncoding = inferHandEncoding(
    num_players, hand_length, num_digits, historyEncoding,
    firstLayer.inSize, hasModernMetadata,
  );

  const config = buildGameConfig(num_players, hand_length, num_digits, {
    maxJump, handEncoding, historyEncoding,
  });
  const weights = extractWeights(params, layers.length, networkType, config);

  return {
    config,
    weights,
    arch: {
      networkType, handEncoding, historyEncoding, maxJump, hiddenLayers: layers,
    },
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface CheckpointData {
  config: GameConfig;
  weights: NetworkWeights;
  arch: CheckpointArch;
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

/** Load checkpoint from an ArrayBuffer (msgpack). */
export function loadCheckpointBytes(buffer: ArrayBuffer): CheckpointData {
  const codec = makeFlaxCodec();
  const state = decode(new Uint8Array(buffer), { extensionCodec: codec }) as Record<string, unknown>;

  const jaxCfg = state['jax_config'] as Record<string, unknown>;
  const layers = normalizeIntList(state['policy_network_layers']);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const paramsTarget = state['params_target'] as any;
  const params = (paramsTarget['params'] ?? paramsTarget) as Record<string, unknown>;

  return buildCheckpoint(jaxCfg, layers, params, {
    network_type: state['network_type'],
    history_encoding: state['history_encoding'],
  });
}

/** Load checkpoint from the JSON export (scripts/export_json.py). */
export function loadCheckpointJson(json: string): CheckpointData {
  const raw = JSON.parse(json) as {
    jax_config: Record<string, unknown>;
    policy_network_layers: number[];
    params_target: Record<string, { kernel: number[][]; bias: number[] }>;
    network_type?: string;
    history_encoding?: string;
  };

  // Convert the nested JSON kernels to the flat Float32Arrays the msgpack path
  // produces, so both formats share the same extraction code.
  const params: Record<string, unknown> = {};
  for (const [name, layer] of Object.entries(raw.params_target)) {
    params[name] = {
      kernel: Float32Array.from(layer.kernel.flatMap(r => r)),
      bias: Float32Array.from(layer.bias),
    };
  }

  return buildCheckpoint(
    raw.jax_config, normalizeIntList(raw.policy_network_layers), params,
    { network_type: raw.network_type, history_encoding: raw.history_encoding },
  );
}
