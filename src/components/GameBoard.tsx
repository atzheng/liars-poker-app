import React, { useState } from 'react';
import type { GameState, GameConfig, NetworkWeights, HistoryEntry } from '../types';
import { legalActionsMask } from '../game';
import type { WinRecord } from '../App';
import BidHistory from './BidHistory';
import ActionPanel from './ActionPanel';

interface Props {
  state: GameState;
  config: GameConfig;
  weights: NetworkWeights;
  history: HistoryEntry[];
  aiThinking: boolean;
  humanPlayer: number;
  record: WinRecord;
  temperature: number;
  onTemperatureChange: (t: number) => void;
  onAction: (action: number) => void;
}

const DIGIT_COLORS = [
  'bg-red-600', 'bg-orange-500', 'bg-yellow-500',
  'bg-green-500', 'bg-blue-500', 'bg-purple-500',
  'bg-pink-500', 'bg-teal-500', 'bg-indigo-500', 'bg-rose-600',
];

function DigitTile({ digit, faceDown = false }: { digit: number; faceDown?: boolean }) {
  if (faceDown) {
    return (
      <div className="w-10 h-10 rounded-lg bg-purple-900 border border-purple-700 flex items-center justify-center text-purple-500 font-bold text-lg select-none">
        ?
      </div>
    );
  }
  return (
    <div
      className={`w-10 h-10 rounded-lg flex items-center justify-center text-white font-bold text-lg shadow select-none ${
        DIGIT_COLORS[(digit - 1) % DIGIT_COLORS.length]
      }`}
    >
      {digit}
    </div>
  );
}

function HandDisplay({
  hand,
  faceDown,
  label,
  labelColor,
  isActive,
}: {
  hand: number[];
  faceDown: boolean;
  label: React.ReactNode;
  labelColor: string;
  isActive: boolean;
}) {
  return (
    <div className={`px-4 py-3 ${isActive ? 'bg-gray-700/40' : ''} rounded-xl transition-colors`}>
      <div className={`text-xs font-semibold mb-2 ${labelColor} flex items-center gap-2`}>
        {label}
        {isActive && (
          <span className="px-1.5 py-0.5 bg-yellow-600/30 text-yellow-400 text-xs rounded-full border border-yellow-700/50">
            thinking…
          </span>
        )}
      </div>
      <div className="flex gap-1.5 flex-wrap">
        {hand.map((digit, i) => (
          <DigitTile key={i} digit={digit} faceDown={faceDown} />
        ))}
      </div>
    </div>
  );
}

export default function GameBoard({ state, config, history, aiThinking, humanPlayer, record, temperature, onTemperatureChange, onAction }: Props) {
  const humanTurn = state.current_player === humanPlayer && !aiThinking;
  const legalMask = humanTurn ? legalActionsMask(state, config) : [];
  const [showAiHand, setShowAiHand] = useState(false);

  const currentBidLabel = (() => {
    if (state.current_bid_action < 0) return 'No bid yet';
    const bidId = state.current_bid_action - 1;
    const number = (bidId % config.num_digits) + 1;  // 1-indexed internally
    const count  = Math.floor(bidId / config.num_digits) + 1;
    return `${count} × ${number}`;
  })();

  const recordStr = `${record.wins}W–${record.losses}L${record.draws > 0 ? `–${record.draws}D` : ''}`;
  const total = record.wins + record.losses + record.draws;

  // Other players (AI-controlled), shown above the history
  const otherPlayers = Array.from({ length: config.num_players }, (_, i) => i).filter(i => i !== humanPlayer);

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col max-w-md mx-auto">
      {/* Header */}
      <div className="px-4 py-3 bg-gray-800 border-b border-gray-700 flex items-center justify-between">
        <h1 className="text-white font-bold text-sm">Liar's Poker</h1>
        <div className="flex items-center gap-3 text-xs text-gray-400">
          <span>{config.num_players}P · {config.hand_length} cards · {config.num_digits} digits</span>
          {total > 0 && (
            <span className="px-2 py-0.5 rounded-full bg-gray-700 text-gray-300 font-medium">
              {recordStr}
            </span>
          )}
          <label className="flex items-center gap-1 text-gray-500">
            <span>T</span>
            <input
              type="number"
              min={0.1}
              max={10}
              step={0.1}
              value={temperature}
              onChange={e => {
                const v = parseFloat(e.target.value);
                if (v > 0) onTemperatureChange(v);
              }}
              className="w-14 bg-gray-700 text-gray-200 rounded px-1.5 py-0.5 text-xs text-right border border-gray-600 focus:border-blue-400 focus:outline-none"
            />
          </label>
        </div>
      </div>

      {/* AI/other player hands */}
      {otherPlayers.map(p => (
        <div key={p} className="bg-gray-800/50 border-b border-gray-700">
          <HandDisplay
            hand={state.hands[p]}
            faceDown={!showAiHand}
            label={
              <span className="flex items-center gap-2">
                {config.num_players === 2 ? 'AI' : `Player ${p}`}
                <button
                  onClick={() => setShowAiHand(v => !v)}
                  className="text-xs px-1.5 py-0.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-400 hover:text-gray-200 transition-colors font-normal"
                >
                  {showAiHand ? 'hide' : 'show'}
                </button>
              </span>
            }
            labelColor="text-purple-400"
            isActive={aiThinking && state.current_player === p}
          />
        </div>
      ))}

      {/* Current bid status */}
      <div className="px-4 py-2 bg-gray-800/30 border-b border-gray-700 flex items-center justify-between">
        <span className="text-xs text-gray-500">Current bid:</span>
        <span className="text-sm font-bold text-yellow-400">{currentBidLabel}</span>
      </div>

      {/* Bid history */}
      <BidHistory history={history} humanPlayer={humanPlayer} numPlayers={config.num_players} config={config} />

      {/* Human hand */}
      <div className="bg-gray-800/50 border-t border-gray-700">
        <HandDisplay
          hand={state.hands[humanPlayer]}
          faceDown={false}
          label="Your hand"
          labelColor="text-blue-400"
          isActive={humanTurn}
        />
      </div>

      {/* Action panel */}
      <div className="bg-gray-800 border-t border-gray-700">
        {humanTurn ? (
          <ActionPanel
            legalMask={legalMask}
            config={config}
            onAction={onAction}
            aiThinking={false}
          />
        ) : (
          <ActionPanel
            legalMask={[]}
            config={config}
            onAction={() => {}}
            aiThinking={aiThinking || state.current_player !== humanPlayer}
          />
        )}
      </div>
    </div>
  );
}
