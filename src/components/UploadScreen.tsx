import React, { useCallback, useState } from 'react';
import { loadCheckpointBytes, loadCheckpointJson, buildGameConfig } from '../checkpoint';
import type { CheckpointData } from '../checkpoint';

interface Props {
  onLoad: (data: CheckpointData, humanPlayer: number) => void;
}

interface ParsedCheckpoint {
  data: CheckpointData;
  numPlayers: number;
  handLength: number;
  numDigits: number;
  humanPlayer: number;
}

export default function UploadScreen({ onLoad }: Props) {
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [parsed, setParsed] = useState<ParsedCheckpoint | null>(null);

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
    const config = buildGameConfig(parsed.numPlayers, parsed.handLength, parsed.numDigits);
    onLoad({ ...parsed.data, config }, parsed.humanPlayer);
  }, [parsed, onLoad]);

  if (parsed) {
    const playerLabels = Array.from({ length: parsed.numPlayers }, (_, i) =>
      i === 0 ? 'P0 (first mover)' : `P${i}`,
    );

    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
        <div className="bg-gray-800 rounded-2xl shadow-2xl p-8 max-w-md w-full">
          <h1 className="text-3xl font-bold text-white mb-1 text-center">Liar's Poker AI</h1>
          <p className="text-gray-400 mb-6 text-sm text-center">Configure game parameters</p>

          <div className="space-y-4 mb-6">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Number of Players</label>
              <input
                type="number"
                min={2}
                max={6}
                value={parsed.numPlayers}
                onChange={e => setParsed(p => {
                  if (!p) return p;
                  const n = Math.max(2, parseInt(e.target.value) || 2);
                  return { ...p, numPlayers: n, humanPlayer: Math.min(p.humanPlayer, n - 1) };
                })}
                className="w-full bg-gray-700 text-white rounded-lg px-3 py-2 border border-gray-600 focus:border-blue-400 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Hand Size (cards per player)</label>
              <input
                type="number"
                min={1}
                max={10}
                value={parsed.handLength}
                onChange={e => setParsed(p => p && ({ ...p, handLength: Math.max(1, parseInt(e.target.value) || 1) }))}
                className="w-full bg-gray-700 text-white rounded-lg px-3 py-2 border border-gray-600 focus:border-blue-400 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Number of Digits (1–N)</label>
              <input
                type="number"
                min={2}
                max={10}
                value={parsed.numDigits}
                onChange={e => setParsed(p => p && ({ ...p, numDigits: Math.max(2, parseInt(e.target.value) || 2) }))}
                className="w-full bg-gray-700 text-white rounded-lg px-3 py-2 border border-gray-600 focus:border-blue-400 focus:outline-none"
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
            <p className="font-medium text-gray-300">Detected from checkpoint:</p>
            <p>
              {parsed.data.config.num_players}p · {parsed.data.config.hand_length} cards ·{' '}
              {parsed.data.config.num_digits} digits · {parsed.data.weights.numLayers} hidden layers
            </p>
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
              className="flex-1 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-500 transition-colors font-medium"
            >
              Start Game
            </button>
          </div>
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
