import React, { useEffect, useRef } from 'react';
import type { HistoryEntry } from '../types';
import { CHALLENGE_ACTION } from '../game';

interface Props {
  history: HistoryEntry[];
  humanPlayer: number;
  numPlayers: number;
}

function playerName(player: number, humanPlayer: number, numPlayers: number): string {
  if (player === humanPlayer) return 'You';
  if (numPlayers === 2) return 'AI';
  return `P${player}`;
}

export default function BidHistory({ history, humanPlayer, numPlayers }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history.length]);

  if (history.length === 0) {
    return (
      <div className="flex-1 overflow-y-auto px-4 py-6 flex items-center justify-center">
        <p className="text-gray-600 text-sm">Game in progress — make the first bid!</p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-3 space-y-2">
      {history.map((entry, idx) => {
        const isHuman = entry.player === humanPlayer;
        return (
          <div
            key={idx}
            className={`flex ${isHuman ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-xs px-3 py-2 rounded-2xl text-sm font-medium shadow ${
                entry.action === CHALLENGE_ACTION
                  ? isHuman
                    ? 'bg-orange-600 text-white rounded-br-sm'
                    : 'bg-orange-800 text-orange-100 rounded-bl-sm'
                  : isHuman
                    ? 'bg-blue-600 text-white rounded-br-sm'
                    : 'bg-gray-700 text-gray-100 rounded-bl-sm'
              }`}
            >
              <span className="opacity-60 text-xs mr-1">{playerName(entry.player, humanPlayer, numPlayers)}:</span>
              {entry.label}
            </div>
          </div>
        );
      })}
      <div ref={bottomRef} />
    </div>
  );
}
