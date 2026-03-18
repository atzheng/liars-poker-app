import React from 'react';
import type { GameState, GameConfig } from '../types';
import { getReturns } from '../game';
import type { WinRecord } from '../App';

interface Props {
  state: GameState;
  config: GameConfig;
  humanPlayer: number;
  record: WinRecord;
  onReplay: () => void;
  onNewCheckpoint: () => void;
}

const DIGIT_COLORS = [
  'bg-red-600', 'bg-orange-500', 'bg-yellow-500',
  'bg-green-500', 'bg-blue-500', 'bg-purple-500',
  'bg-pink-500', 'bg-teal-500', 'bg-indigo-500', 'bg-rose-600',
];

function playerLabel(p: number, humanPlayer: number, numPlayers: number): string {
  if (p === humanPlayer) return 'Your hand';
  if (numPlayers === 2) return "AI's hand";
  return `Player ${p}'s hand`;
}

export default function GameOver({ state, config, humanPlayer, record, onReplay, onNewCheckpoint }: Props) {
  const rewards = getReturns(state, config);
  const humanWon = rewards[humanPlayer] > 0;
  const humanLost = rewards[humanPlayer] < 0;

  const total = record.wins + record.losses + record.draws;
  const winPct = total > 0 ? Math.round((record.wins / total) * 100) : 0;

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
      <div className="bg-gray-800 rounded-2xl shadow-2xl p-8 max-w-lg w-full">
        {/* Result banner */}
        <div className={`text-center mb-4 p-4 rounded-xl ${
          humanWon ? 'bg-green-900/50 border border-green-700' : humanLost ? 'bg-red-900/50 border border-red-700' : 'bg-gray-700/50 border border-gray-600'
        }`}>
          <div className="text-4xl mb-2">{humanWon ? '🎉' : humanLost ? '😔' : '🤝'}</div>
          <h2 className={`text-2xl font-bold ${humanWon ? 'text-green-300' : humanLost ? 'text-red-300' : 'text-gray-300'}`}>
            {humanWon ? 'You Win!' : humanLost ? 'AI Wins!' : 'Draw'}
          </h2>
          <p className="text-gray-400 text-sm mt-1">
            Your score: {rewards[humanPlayer] > 0 ? '+' : ''}{rewards[humanPlayer]}
          </p>
        </div>

        {/* Win record */}
        {total > 0 && (
          <div className="mb-4 px-4 py-3 bg-gray-700/50 rounded-xl flex items-center justify-between">
            <span className="text-xs text-gray-400 font-medium">Session record</span>
            <div className="flex items-center gap-3 text-sm">
              <span className="text-green-400 font-bold">{record.wins}W</span>
              <span className="text-red-400 font-bold">{record.losses}L</span>
              {record.draws > 0 && <span className="text-gray-400 font-bold">{record.draws}D</span>}
              <span className="text-gray-500 text-xs">({winPct}%)</span>
            </div>
          </div>
        )}

        {/* Hands reveal */}
        <div className="space-y-4 mb-6">
          {Array.from({ length: config.num_players }, (_, p) => (
            <div key={p}>
              <div className={`text-sm font-semibold mb-2 ${p === humanPlayer ? 'text-blue-400' : 'text-purple-400'}`}>
                {playerLabel(p, humanPlayer, config.num_players)}
              </div>
              <div className="flex gap-2 flex-wrap">
                {state.hands[p].map((digit, i) => (
                  <div
                    key={i}
                    className={`w-10 h-10 rounded-lg flex items-center justify-center text-white font-bold text-lg shadow ${
                      DIGIT_COLORS[(digit - 1) % DIGIT_COLORS.length]
                    }`}
                  >
                    {digit}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Action buttons */}
        <div className="flex gap-3">
          <button
            onClick={onReplay}
            className="flex-1 py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl transition-colors"
          >
            Play Again
          </button>
          <button
            onClick={onNewCheckpoint}
            className="flex-1 py-3 bg-gray-700 hover:bg-gray-600 text-gray-200 font-medium rounded-xl transition-colors"
          >
            New Checkpoint
          </button>
        </div>
      </div>
    </div>
  );
}
