import React from 'react';
import type { GameConfig } from '../types';

interface Props {
  policy: number[];
  takenAction: number;
  config: GameConfig;
}

/** Interpolate from dark-gray (#1f2937) to bright-blue (#3b82f6) based on t ∈ [0,1]. */
function heatColor(t: number): string {
  const r = Math.round(31  + t * (59  - 31));
  const g = Math.round(41  + t * (130 - 41));
  const b = Math.round(55  + t * (246 - 55));
  return `rgb(${r},${g},${b})`;
}

export default function PolicyHeatmap({ policy, takenAction, config }: Props) {
  const { num_players, hand_length, num_digits } = config;
  const maxCount = hand_length * num_players;

  const challengeProb = policy[0] ?? 0;
  const bidProbs = policy.slice(1);
  const maxProb = Math.max(...policy, 1e-9);

  // Build grid[countIdx][digitIdx] where countIdx = count-1, digitIdx = digit-1
  // action = countIdx * num_digits + digitIdx + 1
  const grid: number[][] = Array.from({ length: maxCount }, (_, ci) =>
    Array.from({ length: num_digits }, (_, di) => policy[ci * num_digits + di + 1] ?? 0),
  );

  // Trim trailing rows that are entirely zero so the heatmap stays compact
  let lastNonZeroRow = -1;
  for (let ci = maxCount - 1; ci >= 0; ci--) {
    if (grid[ci].some(p => p > 1e-5)) { lastNonZeroRow = ci; break; }
  }
  const visibleGrid = grid.slice(0, Math.max(lastNonZeroRow + 1, 3));

  const digits = Array.from({ length: num_digits }, (_, i) => i + 1);

  const CELL_W = Math.min(32, Math.max(24, Math.floor(160 / num_digits)));

  return (
    <div className="mt-1.5 p-2 bg-gray-950 rounded-lg border border-gray-700 text-xs select-none">
      {/* Challenge row */}
      <div className="mb-2">
        <div className="flex items-center gap-1.5">
          <span className="text-gray-500 w-6 shrink-0 text-right">CH</span>
          <div
            className="h-5 rounded flex items-center justify-end px-1.5 font-mono transition-all"
            style={{
              width: `${Math.max(challengeProb / maxProb, 0.04) * 100}%`,
              minWidth: '2.5rem',
              background: heatColor(challengeProb / maxProb),
              color: challengeProb / maxProb > 0.4 ? 'white' : '#9ca3af',
            }}
          >
            {(challengeProb * 100).toFixed(1)}%
          </div>
          {takenAction === 0 && (
            <span className="text-yellow-400 text-xs">◀ took</span>
          )}
        </div>
      </div>

      {/* Digit header */}
      <div className="flex gap-px mb-0.5" style={{ marginLeft: '1.75rem' }}>
        {digits.map(d => (
          <div
            key={d}
            className="text-center text-gray-500 font-medium"
            style={{ width: CELL_W }}
          >
            {d}
          </div>
        ))}
      </div>

      {/* Bid grid */}
      {visibleGrid.map((row, ci) => {
        const count = ci + 1;
        return (
          <div key={ci} className="flex items-center gap-px mb-px">
            <div className="text-gray-500 text-right mr-1 shrink-0" style={{ width: '1.5rem' }}>
              {count}×
            </div>
            {row.map((p, di) => {
              const action = ci * num_digits + di + 1;
              const isTaken = action === takenAction;
              const t = p / maxProb;
              return (
                <div
                  key={di}
                  className="flex items-center justify-center rounded-sm font-mono transition-colors"
                  style={{
                    width: CELL_W,
                    height: 20,
                    background: heatColor(t),
                    color: t > 0.45 ? 'white' : '#6b7280',
                    outline: isTaken ? '2px solid #facc15' : 'none',
                    outlineOffset: '-1px',
                  }}
                  title={`${count} × ${di + 1}: ${(p * 100).toFixed(2)}%`}
                >
                  {p > 0.005 ? `${Math.round(p * 100)}` : ''}
                </div>
              );
            })}
          </div>
        );
      })}

      <div className="mt-1.5 text-gray-600 text-xs">
        Numbers show % · yellow outline = chosen action
      </div>
    </div>
  );
}
