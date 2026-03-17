import React, { useState, useEffect } from 'react';
import type { GameConfig } from '../types';

interface Props {
  legalMask: boolean[];
  config: GameConfig;
  onAction: (action: number) => void;
  aiThinking: boolean;
}

function getBidAction(count: number, digit: number, config: GameConfig): number {
  return (count - 1) * config.num_digits + (digit - 1) + 1; // +1 for BID_ACTION_OFFSET
}

function legalCountsForDigit(digit: number, legalMask: boolean[], config: GameConfig): number[] {
  const maxCount = config.hand_length * config.num_players;
  const result: number[] = [];
  for (let c = 1; c <= maxCount; c++) {
    const action = getBidAction(c, digit, config);
    if (action < legalMask.length && legalMask[action]) result.push(c);
  }
  return result;
}

function hasAnyLegalBidForDigit(digit: number, legalMask: boolean[], config: GameConfig): boolean {
  return legalCountsForDigit(digit, legalMask, config).length > 0;
}

export default function ActionPanel({ legalMask, config, onAction, aiThinking }: Props) {
  const [selectedDigit, setSelectedDigit] = useState<number | null>(null);

  // Clear digit selection when legal mask changes (new game state)
  useEffect(() => {
    setSelectedDigit(null);
  }, [legalMask]);

  if (aiThinking) {
    return (
      <div className="p-4 flex items-center justify-center gap-2 text-gray-400">
        <span className="animate-spin inline-block text-lg">⟳</span>
        <span className="text-sm">AI is thinking…</span>
      </div>
    );
  }

  const challengeLegal = legalMask.length > 0 && legalMask[0];
  const digits = Array.from({ length: config.num_digits }, (_, i) => i + 1);

  return (
    <div className="p-3 space-y-3">
      {/* Challenge button */}
      <button
        onClick={() => onAction(0)}
        disabled={!challengeLegal}
        className={`w-full py-2 rounded-xl font-bold text-sm transition-all ${
          challengeLegal
            ? 'bg-orange-600 hover:bg-orange-500 text-white shadow-lg hover:shadow-orange-500/30'
            : 'bg-gray-800 text-gray-600 cursor-not-allowed'
        }`}
      >
        CHALLENGE
      </button>

      {selectedDigit === null ? (
        /* Step 1: choose digit */
        <div>
          <p className="text-xs text-gray-500 mb-2">Bid on digit:</p>
          <div className="flex flex-wrap gap-2">
            {digits.map(d => {
              const legal = hasAnyLegalBidForDigit(d, legalMask, config);
              return (
                <button
                  key={d}
                  onClick={() => setSelectedDigit(d)}
                  disabled={!legal}
                  className={`w-10 h-10 rounded-lg font-bold text-sm transition-all ${
                    legal
                      ? 'bg-blue-700 hover:bg-blue-500 text-white shadow'
                      : 'bg-gray-800 text-gray-600 cursor-not-allowed'
                  }`}
                >
                  {d}
                </button>
              );
            })}
          </div>
        </div>
      ) : (
        /* Step 2: choose count for selected digit */
        <div>
          <div className="flex items-center gap-2 mb-2">
            <button
              onClick={() => setSelectedDigit(null)}
              className="text-xs text-gray-400 hover:text-gray-200 transition-colors px-2 py-1 rounded bg-gray-700 hover:bg-gray-600"
            >
              ← back
            </button>
            <p className="text-xs text-gray-400">
              How many <span className="text-white font-bold">{selectedDigit}</span>s?
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {legalCountsForDigit(selectedDigit, legalMask, config).map(c => (
              <button
                key={c}
                onClick={() => {
                  onAction(getBidAction(c, selectedDigit, config));
                  setSelectedDigit(null);
                }}
                className="px-3 h-10 rounded-lg font-bold text-sm bg-blue-700 hover:bg-blue-500 text-white shadow transition-all"
              >
                {c}×
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
