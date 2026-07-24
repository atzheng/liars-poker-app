import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { CheckpointData } from './checkpoint';
import type { GameConfig, GameState, HistoryEntry, NetworkWeights } from './types';
import { applyAction, CHALLENGE_ACTION, decodeBid, dealGame, getReturns, isTerminal, legalActionsMask } from './game';
import { chooseAiAction } from './agent';
import { chooseServerAction } from './serverAgent';
import { localPolicySource, serverPolicySource } from './policySource';
import UploadScreen from './components/UploadScreen';
import GameBoard from './components/GameBoard';
import PolicyExplorer, { type ExplorerInit } from './components/PolicyExplorer';
import { handToCounts } from './trajectory';

type Phase = 'upload' | 'game' | 'explorer';

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
  // When set, the AI moves are computed by the Python transformer backend at
  // this base URL instead of the in-browser TS-MLP (weights).
  const [serverUrl, setServerUrl] = useState<string | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [aiThinking, setAiThinking] = useState(false);
  const [humanPlayer, setHumanPlayer] = useState(1);
  const [record, setRecord] = useState<WinRecord>({ wins: 0, losses: 0, draws: 0 });
  const [temperature, setTemperature] = useState(1);
  const [policyThreshold, setPolicyThreshold] = useState(0);
  const [explorerLabel, setExplorerLabel] = useState<string>('');
  // Pre-loaded trajectory for the Explorer (set when launched from the game via
  // "Inspect in Policy Explorer"; undefined => fresh/standalone Explorer).
  const [explorerInit, setExplorerInit] = useState<ExplorerInit | undefined>(undefined);
  // Where "Back" returns from the Explorer: 'upload' (standalone) or 'game' (inspect).
  const [explorerReturn, setExplorerReturn] = useState<Phase>('upload');

  const aiScheduled = useRef(false);

  const startGame = useCallback(
    (cfg: GameConfig, w: NetworkWeights | null, hp: number, srv: string | null) => {
      const state = dealGame(cfg);
      setConfig(cfg);
      setWeights(w);
      setServerUrl(srv);
      setHumanPlayer(hp);
      setGameState(state);
      setHistory([]);
      setAiThinking(false);
      aiScheduled.current = false;
      recordUpdated.current = false;
      setPhase('game');
    },
    [],
  );

  const handleLoad = useCallback((data: CheckpointData, hp: number) => {
    startGame(data.config, data.weights, hp, null);
  }, [startGame]);

  const handleConnectServer = useCallback((cfg: GameConfig, hp: number, url: string) => {
    startGame(cfg, null, hp, url);
  }, [startGame]);

  // Standalone Explorer from the upload screen — against the connected server
  // or an in-browser checkpoint, whichever was loaded.
  const handleOpenExplorer = useCallback(
    (cfg: GameConfig, agent: { serverUrl?: string; weights?: NetworkWeights }, label: string) => {
      setConfig(cfg);
      setServerUrl(agent.serverUrl ?? null);
      setWeights(agent.weights ?? null);
      setExplorerLabel(label);
      setExplorerInit(undefined);
      setExplorerReturn('upload');
      setPhase('explorer');
    },
    [],
  );

  // How the Explorer evaluates states: the server when connected, otherwise the
  // in-browser network. Memoized so the Explorer's query effect is stable.
  const explorerQuery = useMemo(() => {
    if (!config) return null;
    if (serverUrl) return serverPolicySource(serverUrl, config);
    if (weights) return localPolicySource(weights, config);
    return null;
  }, [config, serverUrl, weights]);

  // "Inspect in Policy Explorer" from the live game: capture the chosen seat's
  // ACTUAL dealt hand + the full bid/challenge history so far, and pre-load the
  // Explorer with it, viewing the trajectory from that seat's perspective. Every
  // seat is inspectable (the human's own, and each AI opponent), in both
  // backends — the Explorer evaluates through explorerQuery.
  const handleInspect = useCallback((seat: number) => {
    if (!gameState || !config || !(serverUrl || weights)) return;
    const handCounts = handToCounts(gameState.hands[seat], config.num_digits);
    const sequence = history.map(h => h.action);
    setExplorerInit({ actingSeat: seat, handCounts, sequence });
    setExplorerLabel(`from current game · P${seat}`);
    setExplorerReturn('game');
    setPhase('explorer');
  }, [gameState, config, serverUrl, weights, history]);

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

  // AI turn effect: fires whenever it's not the human's turn.
  // Two backends: in-browser TS-MLP (weights) or the Python transformer server
  // (serverUrl). Exactly one of the two is set for a given game session.
  useEffect(() => {
    if (phase !== 'game') return;
    if (!gameState || !config) return;
    if (!weights && !serverUrl) return;
    if (isTerminal(gameState)) return;
    if (gameState.current_player === humanPlayer) return;
    if (aiScheduled.current) return;

    aiScheduled.current = true;
    setAiThinking(true);
    let cancelled = false;

    const delay = 400 + Math.random() * 400;
    const timer = setTimeout(async () => {
      try {
        if (serverUrl) {
          const { action, policy } = await chooseServerAction(
            serverUrl, gameState, config, { temperature, greedy: false, threshold: policyThreshold });
          if (cancelled) return;
          applyPlayerAction(action, policy);
        } else if (weights) {
          const { action, policy } = chooseAiAction(
            gameState, config, weights, temperature, policyThreshold);
          if (cancelled) return;
          applyPlayerAction(action, policy);
        }
      } catch (e) {
        // eslint-disable-next-line no-console
        console.error('AI move failed:', e);
      } finally {
        if (!cancelled) {
          setAiThinking(false);
          aiScheduled.current = false;
        }
      }
    }, delay);

    return () => {
      cancelled = true;
      clearTimeout(timer);
      aiScheduled.current = false;
      setAiThinking(false);
    };
  }, [gameState, config, weights, serverUrl, phase, humanPlayer, temperature, policyThreshold, applyPlayerAction]);

  // Terminal detection: update record
  const recordUpdated = useRef(false);
  useEffect(() => {
    if (phase !== 'game' || !gameState || !config) return;
    if (!isTerminal(gameState)) return;
    if (recordUpdated.current) return;
    recordUpdated.current = true;
    const rewards = getReturns(gameState, config);
    const r = rewards[humanPlayer];
    const t = setTimeout(() => {
      setRecord(rec => ({
        wins:   rec.wins   + (r > 0 ? 1 : 0),
        losses: rec.losses + (r < 0 ? 1 : 0),
        draws:  rec.draws  + (r === 0 ? 1 : 0),
      }));
    }, 600);
    return () => clearTimeout(t);
  }, [gameState, phase, humanPlayer, config]);

  if (phase === 'upload') {
    return (
      <UploadScreen
        onLoad={handleLoad}
        onConnectServer={handleConnectServer}
        onOpenExplorer={handleOpenExplorer}
      />
    );
  }

  if (phase === 'explorer' && config && explorerQuery) {
    return (
      <PolicyExplorer
        config={config}
        query={explorerQuery}
        agentLabel={explorerLabel}
        initial={explorerInit}
        backLabel={explorerReturn === 'game' ? 'Back to game' : 'Back'}
        onBack={() => setPhase(explorerReturn)}
      />
    );
  }

  if (phase === 'game' && gameState && config && (weights || serverUrl)) {
    return (
      <GameBoard
        state={gameState}
        config={config}
        weights={weights}
        agentLabel={serverUrl ? 'Transformer AI (server)' : 'MLP AI (in-browser)'}
        history={history}
        aiThinking={aiThinking}
        humanPlayer={humanPlayer}
        record={record}
        temperature={temperature}
        onTemperatureChange={setTemperature}
        policyThreshold={policyThreshold}
        onPolicyThresholdChange={setPolicyThreshold}
        onAction={handleHumanAction}
        onInspect={handleInspect}
        onReplay={() => startGame(config, weights, humanPlayer, serverUrl)}
        onNewCheckpoint={() => {
          setPhase('upload');
          setGameState(null);
        }}
      />
    );
  }

  return null;
}
