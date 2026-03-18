import React, { useCallback, useEffect, useRef, useState } from 'react';
import type { CheckpointData } from './checkpoint';
import type { GameConfig, GameState, HistoryEntry, NetworkWeights } from './types';
import { applyAction, CHALLENGE_ACTION, decodeBid, dealGame, getReturns, isTerminal, legalActionsMask } from './game';
import { chooseAiAction } from './agent';
import UploadScreen from './components/UploadScreen';
import GameBoard from './components/GameBoard';
import GameOver from './components/GameOver';

type Phase = 'upload' | 'game' | 'gameover';

export interface WinRecord {
  wins: number;
  losses: number;
  draws: number;
}

function makeHistoryEntry(
  action: number,
  player: number,
  config: GameConfig,
  policy?: number[],
): HistoryEntry {
  if (action === CHALLENGE_ACTION) {
    return { type: 'challenge', player, action, label: 'CHALLENGE!', policy };
  }
  const bidId = action - 1;
  const decoded = decodeBid(bidId, config);
  return {
    type: 'bid',
    player,
    action,
    decodedBid: decoded,
    label: `${decoded.count} × ${decoded.number}`,
    policy,
  };
}

export default function App() {
  const [phase, setPhase] = useState<Phase>('upload');
  const [config, setConfig] = useState<GameConfig | null>(null);
  const [weights, setWeights] = useState<NetworkWeights | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [aiThinking, setAiThinking] = useState(false);
  const [humanPlayer, setHumanPlayer] = useState(1);
  const [record, setRecord] = useState<WinRecord>({ wins: 0, losses: 0, draws: 0 });
  const [temperature, setTemperature] = useState(1);

  const aiScheduled = useRef(false);

  const startGame = useCallback((cfg: GameConfig, w: NetworkWeights, hp: number) => {
    const state = dealGame(cfg);
    setConfig(cfg);
    setWeights(w);
    setHumanPlayer(hp);
    setGameState(state);
    setHistory([]);
    setAiThinking(false);
    aiScheduled.current = false;
    setPhase('game');
  }, []);

  const handleLoad = useCallback((data: CheckpointData, hp: number) => {
    startGame(data.config, data.weights, hp);
  }, [startGame]);

  const applyPlayerAction = useCallback(
    (action: number, policy?: number[]) => {
      if (!gameState || !config) return;
      const legal = legalActionsMask(gameState, config);
      if (!legal[action]) return;
      const entry = makeHistoryEntry(action, gameState.current_player, config, policy);
      const next = applyAction(gameState, config, action);
      setGameState(next);
      setHistory(h => [...h, entry]);
    },
    [gameState, config],
  );

  const handleHumanAction = useCallback(
    (action: number) => {
      if (!gameState || !config || gameState.current_player !== humanPlayer || aiThinking) return;
      applyPlayerAction(action);
    },
    [gameState, config, humanPlayer, aiThinking, applyPlayerAction],
  );

  // AI turn effect: fires whenever it's not the human's turn
  useEffect(() => {
    if (phase !== 'game') return;
    if (!gameState || !config || !weights) return;
    if (isTerminal(gameState)) return;
    if (gameState.current_player === humanPlayer) return;
    if (aiScheduled.current) return;

    aiScheduled.current = true;
    setAiThinking(true);

    const delay = 400 + Math.random() * 400;
    const timer = setTimeout(() => {
      const { action, policy } = chooseAiAction(gameState, config, weights, temperature);
      applyPlayerAction(action, policy);
      setAiThinking(false);
      aiScheduled.current = false;
    }, delay);

    return () => {
      clearTimeout(timer);
      aiScheduled.current = false;
      setAiThinking(false);
    };
  }, [gameState, config, weights, phase, humanPlayer, applyPlayerAction]);

  // Terminal detection: update record and switch to gameover
  useEffect(() => {
    if (phase !== 'game' || !gameState || !config) return;
    if (!isTerminal(gameState)) return;
    const rewards = getReturns(gameState, config);
    const r = rewards[humanPlayer];
    const t = setTimeout(() => {
      setRecord(rec => ({
        wins:   rec.wins   + (r > 0 ? 1 : 0),
        losses: rec.losses + (r < 0 ? 1 : 0),
        draws:  rec.draws  + (r === 0 ? 1 : 0),
      }));
      setPhase('gameover');
    }, 600);
    return () => clearTimeout(t);
  }, [gameState, phase, humanPlayer, config]);

  if (phase === 'upload') {
    return <UploadScreen onLoad={handleLoad} />;
  }

  if (phase === 'gameover' && gameState && config) {
    return (
      <GameOver
        state={gameState}
        config={config}
        humanPlayer={humanPlayer}
        record={record}
        onReplay={() => startGame(config, weights!, humanPlayer)}
        onNewCheckpoint={() => {
          setPhase('upload');
          setGameState(null);
        }}
      />
    );
  }

  if (phase === 'game' && gameState && config && weights) {
    return (
      <GameBoard
        state={gameState}
        config={config}
        weights={weights}
        history={history}
        aiThinking={aiThinking}
        humanPlayer={humanPlayer}
        record={record}
        temperature={temperature}
        onTemperatureChange={setTemperature}
        onAction={handleHumanAction}
      />
    );
  }

  return null;
}
