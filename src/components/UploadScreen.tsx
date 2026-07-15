import React, { useCallback, useState } from 'react';
import { loadCheckpointBytes, loadCheckpointJson, buildGameConfig } from '../checkpoint';
import type { CheckpointData } from '../checkpoint';
import type { GameConfig } from '../types';
import { fetchServerConfig, fetchCheckpoints, loadServerCheckpoint } from '../serverAgent';
import type { ServerInfo } from '../serverAgent';

const BUILT_IN_AGENTS: { name: string; description: string; path: string }[] = [
  { name: '3×3', description: '3 players · 3 cards', path: '/agents/3x3.msgpack' },
];

const DEFAULT_SERVER_URL = 'http://localhost:8000';

interface Props {
  onLoad: (data: CheckpointData, humanPlayer: number) => void;
  onConnectServer: (config: GameConfig, humanPlayer: number, url: string) => void;
  onOpenExplorer: (config: GameConfig, url: string, label: string) => void;
}

interface ParsedCheckpoint {
  data: CheckpointData | null;   // null in server mode
  serverUrl?: string;            // set → Transformer AI (server) mode
  serverInfo?: string;           // description of the loaded server checkpoint
  serverLoaded?: boolean;        // server has a checkpoint loaded (dims valid)
  checkpointPath?: string | null; // currently-loaded checkpoint path (server)
  checkpointList?: string[];     // checkpoints the server offers to load
  browseDir?: string | null;     // dir/prefix the list was fetched from
  numPlayers: number;
  handLength: number;
  numDigits: number;
  maxJump?: number | null;       // action-space abstraction (server mode)
  humanPlayer: number;
}

/** Short label for a checkpoint path: "<run>/agent_NNNN.msgpack". */
function checkpointLabel(path: string): string {
  const parts = path.replace(/\/+$/, '').split('/');
  return parts.slice(-2).join('/');
}

/** Build the parsed-state fields from a freshly-fetched/loaded ServerInfo. */
function serverInfoToParsed(info: ServerInfo): Partial<ParsedCheckpoint> {
  if (!info.loaded || !info.config) {
    return {
      serverLoaded: false,
      serverInfo: 'no checkpoint loaded',
      checkpointPath: null,
    };
  }
  return {
    serverLoaded: true,
    serverInfo: `${info.network_type} · ${info.checkpoint}`
      + (info.maxJump != null ? ` · max_jump=${info.maxJump}` : ''),
    checkpointPath: info.checkpointPath ?? null,
    numPlayers: info.config.num_players,
    handLength: info.config.hand_length,
    numDigits: info.config.num_digits,
    maxJump: info.maxJump,
  };
}

export default function UploadScreen({ onLoad, onConnectServer, onOpenExplorer }: Props) {
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [parsed, setParsed] = useState<ParsedCheckpoint | null>(null);
  const [serverUrl, setServerUrl] = useState(DEFAULT_SERVER_URL);
  // Checkpoint path typed/pasted into the server picker (full path or s3://).
  const [ckptInput, setCkptInput] = useState('');

  const handleConnect = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const info = await fetchServerConfig(serverUrl);
      // Best-effort list of checkpoints the server can hot-load (mlp backend);
      // older backends 404 → empty list and the picker is simply hidden.
      let checkpointList: string[] = [];
      let browseDir: string | null = info.defaultDir ?? null;
      try {
        const cp = await fetchCheckpoints(serverUrl);
        checkpointList = cp.checkpoints;
        browseDir = cp.dir ?? browseDir;
      } catch { /* listing optional */ }
      // Prefill the path box with the already-loaded checkpoint (if any) so the
      // user can see/edit it; otherwise leave it blank to paste one.
      setCkptInput(info.checkpointPath ?? '');
      setParsed({
        data: null,
        serverUrl,
        checkpointList,
        browseDir,
        // dims default to sensible fallbacks until a checkpoint is loaded.
        numPlayers: info.config?.num_players ?? 2,
        handLength: info.config?.hand_length ?? 1,
        numDigits: info.config?.num_digits ?? 2,
        humanPlayer: 1,
        ...serverInfoToParsed(info),
      });
    } catch (e) {
      setError(
        `Could not reach transformer server at ${serverUrl}: ` +
        `${e instanceof Error ? e.message : String(e)}. ` +
        `Is the inference server running?`,
      );
    } finally {
      setLoading(false);
    }
  }, [serverUrl]);

  const handleLoadCheckpoint = useCallback(async (checkpoint: string) => {
    const path = checkpoint.trim();
    if (!parsed?.serverUrl || !path) return;
    setLoading(true);
    setError(null);
    try {
      const info = await loadServerCheckpoint(parsed.serverUrl, path);
      setCkptInput(info.checkpointPath ?? path);
      setParsed(p => p && ({ ...p, ...serverInfoToParsed(info) }));
    } catch (e) {
      setError(`Failed to load checkpoint: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setLoading(false);
    }
  }, [parsed?.serverUrl]);

  const handleRefreshList = useCallback(async (dir: string) => {
    if (!parsed?.serverUrl) return;
    setLoading(true);
    setError(null);
    try {
      const cp = await fetchCheckpoints(parsed.serverUrl, dir || undefined);
      setParsed(p => p && ({ ...p, checkpointList: cp.checkpoints, browseDir: cp.dir ?? dir }));
    } catch (e) {
      setError(`Failed to list checkpoints: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setLoading(false);
    }
  }, [parsed?.serverUrl]);

  const handlePreset = useCallback(async (path: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(path);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const buf = await res.arrayBuffer();
      const data = loadCheckpointBytes(buf);
      setParsed({
        data,
        numPlayers: data.config.num_players,
        handLength: data.config.hand_length,
        numDigits: data.config.num_digits,
        humanPlayer: 1,
      });
    } catch (e) {
      setError(`Failed to load agent: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleFile = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      let data: CheckpointData;
      if (file.name.endsWith('.json')) {
        const text = await file.text();
        data = loadCheckpointJson(text);
      } else {
        const buf = await file.arrayBuffer();
        data = loadCheckpointBytes(buf);
      }
      setParsed({
        data,
        numPlayers: data.config.num_players,
        handLength: data.config.hand_length,
        numDigits: data.config.num_digits,
        humanPlayer: 1,
      });
    } catch (e) {
      setError(`Failed to load checkpoint: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const handleStart = useCallback(() => {
    if (!parsed) return;
    const config = buildGameConfig(parsed.numPlayers, parsed.handLength, parsed.numDigits, parsed.maxJump);
    if (parsed.serverUrl) {
      onConnectServer(config, parsed.humanPlayer, parsed.serverUrl);
    } else if (parsed.data) {
      onLoad({ ...parsed.data, config }, parsed.humanPlayer);
    }
  }, [parsed, onLoad, onConnectServer]);

  const handleExplorer = useCallback(() => {
    if (!parsed || !parsed.serverUrl) return;
    const config = buildGameConfig(parsed.numPlayers, parsed.handLength, parsed.numDigits, parsed.maxJump);
    onOpenExplorer(config, parsed.serverUrl, parsed.serverInfo ?? parsed.serverUrl);
  }, [parsed, onOpenExplorer]);

  if (parsed) {
    const isServer = !!parsed.serverUrl;
    const playerLabels = Array.from({ length: parsed.numPlayers }, (_, i) =>
      i === 0 ? 'P0 (first mover)' : `P${i}`,
    );

    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
        <div className="bg-gray-800 rounded-2xl shadow-2xl p-8 max-w-md w-full">
          <h1 className="text-3xl font-bold text-white mb-1 text-center">Liar's Poker AI</h1>
          <p className="text-gray-400 mb-6 text-sm text-center">
            {isServer ? 'AI (server) — dims fixed by checkpoint' : 'Configure game parameters'}
          </p>

          {isServer && (
            <div className="mb-6 bg-gray-700/40 rounded-lg p-3 space-y-2">
              <label className="block text-sm text-gray-300 font-medium">Checkpoint</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={ckptInput}
                  placeholder="path or s3:// to a .msgpack checkpoint"
                  disabled={loading}
                  onChange={e => setCkptInput(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') handleLoadCheckpoint(ckptInput); }}
                  className="flex-1 bg-gray-700 text-white rounded-lg px-2 py-2 border border-gray-600 focus:border-purple-400 focus:outline-none text-sm disabled:opacity-50"
                />
                <button
                  onClick={() => handleLoadCheckpoint(ckptInput)}
                  disabled={loading || !ckptInput.trim()}
                  className="px-4 py-2 rounded-lg bg-purple-600 text-white hover:bg-purple-500 transition-colors text-sm disabled:opacity-50"
                >
                  {loading ? '…' : 'Load'}
                </button>
              </div>

              {parsed.checkpointList && parsed.checkpointList.length > 0 && (
                <div className="flex gap-2">
                  <select
                    value=""
                    disabled={loading}
                    onChange={e => { if (e.target.value) { setCkptInput(e.target.value); handleLoadCheckpoint(e.target.value); } }}
                    className="flex-1 bg-gray-700 text-white rounded-lg px-2 py-2 border border-gray-600 focus:border-purple-400 focus:outline-none text-sm disabled:opacity-50"
                  >
                    <option value="">— or pick a discovered checkpoint —</option>
                    {parsed.checkpointList.map(path => (
                      <option key={path} value={path}>{checkpointLabel(path)}</option>
                    ))}
                  </select>
                  <button
                    onClick={() => handleRefreshList(parsed.browseDir ?? '')}
                    disabled={loading}
                    title={`Re-list ${parsed.browseDir ?? 'default dir'}`}
                    className="px-3 py-2 rounded-lg bg-gray-600 text-white hover:bg-gray-500 transition-colors text-sm disabled:opacity-50"
                  >
                    ⟳
                  </button>
                </div>
              )}

              <p className="text-gray-500 text-xs">
                {parsed.serverLoaded
                  ? `Loaded: ${parsed.checkpointPath ?? '—'}`
                  : 'Paste a checkpoint path and click Load — game parameters populate from it.'}
              </p>
            </div>
          )}

          <div className="space-y-4 mb-6">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Number of Players</label>
              <input
                type="number"
                min={2}
                max={6}
                value={parsed.numPlayers}
                disabled={isServer}
                onChange={e => setParsed(p => {
                  if (!p) return p;
                  const n = Math.max(2, parseInt(e.target.value) || 2);
                  return { ...p, numPlayers: n, humanPlayer: Math.min(p.humanPlayer, n - 1) };
                })}
                className="w-full bg-gray-700 text-white rounded-lg px-3 py-2 border border-gray-600 focus:border-blue-400 focus:outline-none disabled:opacity-50"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Hand Size (cards per player)</label>
              <input
                type="number"
                min={1}
                max={10}
                value={parsed.handLength}
                disabled={isServer}
                onChange={e => setParsed(p => p && ({ ...p, handLength: Math.max(1, parseInt(e.target.value) || 1) }))}
                className="w-full bg-gray-700 text-white rounded-lg px-3 py-2 border border-gray-600 focus:border-blue-400 focus:outline-none disabled:opacity-50"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Number of Digits (1–N)</label>
              <input
                type="number"
                min={2}
                max={10}
                value={parsed.numDigits}
                disabled={isServer}
                onChange={e => setParsed(p => p && ({ ...p, numDigits: Math.max(2, parseInt(e.target.value) || 2) }))}
                className="w-full bg-gray-700 text-white rounded-lg px-3 py-2 border border-gray-600 focus:border-blue-400 focus:outline-none disabled:opacity-50"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">Play as</label>
              <div className="flex flex-wrap gap-2">
                {playerLabels.map((label, i) => (
                  <button
                    key={i}
                    onClick={() => setParsed(p => p && ({ ...p, humanPlayer: i }))}
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                      parsed.humanPlayer === i
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="bg-gray-700/50 rounded-lg p-3 mb-6 text-xs text-gray-400 space-y-1">
            <p className="font-medium text-gray-300">
              {isServer ? 'Transformer server:' : 'Detected from checkpoint:'}
            </p>
            {isServer ? (
              parsed.serverLoaded ? (
                <p>
                  {parsed.numPlayers}p · {parsed.handLength} cards · {parsed.numDigits} digits ·{' '}
                  {parsed.serverInfo} @ {parsed.serverUrl}
                </p>
              ) : (
                <p>No checkpoint loaded @ {parsed.serverUrl} — choose one above to begin.</p>
              )
            ) : (
              <p>
                {parsed.data!.config.num_players}p · {parsed.data!.config.hand_length} cards ·{' '}
                {parsed.data!.config.num_digits} digits · {parsed.data!.weights.numLayers} hidden layers
              </p>
            )}
          </div>

          <div className="flex gap-3">
            <button
              onClick={() => { setParsed(null); setError(null); }}
              className="flex-1 py-2 rounded-lg bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors text-sm"
            >
              Back
            </button>
            <button
              onClick={handleStart}
              disabled={isServer && !parsed.serverLoaded}
              className="flex-1 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-500 transition-colors font-medium disabled:opacity-50 disabled:hover:bg-blue-600"
            >
              Start Game
            </button>
          </div>

          {isServer && parsed.serverLoaded && (
            <button
              onClick={handleExplorer}
              className="w-full mt-3 py-2 rounded-lg bg-purple-700 text-white hover:bg-purple-600 transition-colors font-medium"
            >
              Policy Explorer (analysis)
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
      <div className="bg-gray-800 rounded-2xl shadow-2xl p-8 max-w-md w-full text-center">
        <h1 className="text-3xl font-bold text-white mb-2">Liar's Poker AI</h1>
        <p className="text-gray-400 mb-8 text-sm">
          Play against a trained RNaD agent in your browser.
        </p>

        <div className="mb-6">
          <p className="text-gray-400 text-sm mb-3 text-left">Play the transformer (server)</p>
          <div className="flex gap-2">
            <input
              type="text"
              value={serverUrl}
              onChange={e => setServerUrl(e.target.value)}
              placeholder="http://localhost:8000"
              className="flex-1 bg-gray-700 text-white rounded-lg px-3 py-2 border border-gray-600 focus:border-purple-400 focus:outline-none text-sm"
            />
            <button
              onClick={handleConnect}
              disabled={loading}
              className="px-4 py-2 rounded-lg bg-purple-600 text-white hover:bg-purple-500 transition-colors font-medium text-sm disabled:opacity-50"
            >
              {loading ? '…' : 'Connect'}
            </button>
          </div>
          <p className="text-gray-600 text-xs mt-2 text-left">
            Runs the real lpt agent via <code className="bg-gray-700 px-1 rounded">serve_agent.py</code>.
          </p>
        </div>

        <div className="flex items-center gap-3 mb-6">
          <div className="flex-1 h-px bg-gray-700" />
          <span className="text-gray-500 text-xs">or in-browser MLP</span>
          <div className="flex-1 h-px bg-gray-700" />
        </div>

        <div className="mb-6">
          <p className="text-gray-400 text-sm mb-3 text-left">Choose an agent</p>
          <div className="flex flex-col gap-2">
            {BUILT_IN_AGENTS.map(agent => (
              <button
                key={agent.path}
                onClick={() => handlePreset(agent.path)}
                disabled={loading}
                className="flex items-center justify-between px-4 py-3 bg-gray-700 hover:bg-gray-600 rounded-xl text-left transition-colors disabled:opacity-50"
              >
                <span className="text-white font-medium">{agent.name}</span>
                <span className="text-gray-400 text-sm">{agent.description}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-3 mb-6">
          <div className="flex-1 h-px bg-gray-700" />
          <span className="text-gray-500 text-xs">or upload your own</span>
          <div className="flex-1 h-px bg-gray-700" />
        </div>

        <label
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`flex flex-col items-center justify-center border-2 border-dashed rounded-xl p-10 cursor-pointer transition-colors ${
            dragging
              ? 'border-blue-400 bg-blue-900/20'
              : 'border-gray-600 hover:border-gray-400 hover:bg-gray-700/30'
          }`}
        >
          <div className="text-5xl mb-4">📂</div>
          <p className="text-gray-300 font-medium">
            {loading ? 'Loading...' : 'Drop checkpoint here'}
          </p>
          <p className="text-gray-500 text-xs mt-1">or click to browse</p>
          <p className="text-gray-600 text-xs mt-3">.msgpack or .json</p>
          <input
            type="file"
            accept=".msgpack,.json"
            className="hidden"
            onChange={onInputChange}
            disabled={loading}
          />
        </label>

        {error && (
          <div className="mt-4 p-3 bg-red-900/40 border border-red-700 rounded-lg text-red-300 text-sm text-left">
            {error}
          </div>
        )}

        <div className="mt-6 text-gray-600 text-xs text-left space-y-1">
          <p className="font-medium text-gray-500">To get a checkpoint:</p>
          <p>1. Convert pickle: <code className="bg-gray-700 px-1 rounded">python convert_checkpoint.py agent_N.pickle</code></p>
          <p>2. Or export JSON: <code className="bg-gray-700 px-1 rounded">python scripts/export_json.py agent_N.pickle</code></p>
        </div>
      </div>
    </div>
  );
}
