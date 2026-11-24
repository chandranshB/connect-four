import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Check, Plus } from 'lucide-react';
import { Connect4AIClient, type ModelTier } from './engine/aiClient';
import { COLS, canPlay, replayMoves, type Player } from './engine/c4';
import { BoardArena } from './components/BoardArena';

interface DifficultyOption {
  id: ModelTier;
  label: string;
  sims: number;
}

const DIFFICULTIES: DifficultyOption[] = [
  { id: 'early', label: 'Easy', sims: 0 },
  { id: 'mid', label: 'Medium', sims: 32 },
  { id: 'pro', label: 'Pro', sims: 96 },
];

export function App() {
  const clientRef = useRef<Connect4AIClient | null>(null);
  if (!clientRef.current) {
    clientRef.current = new Connect4AIClient();
  }

  const [tier, setTier] = useState<ModelTier>('pro');
  const [humanSide, setHumanSide] = useState<Player>(1); // 1 = Red (1st), -1 = Yellow (2nd)
  const [moves, setMoves] = useState<{ col: number; player: Player }[]>([]);
  const [resetCount, setResetCount] = useState<number>(0);

  const [hintCol, setHintCol] = useState<number | null>(null);
  const [isThinking, setIsThinking] = useState<boolean>(false);
  const [isHintPending, setIsHintPending] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [reloadNonce, setReloadNonce] = useState<number>(0);

  const activeDifficulty = DIFFICULTIES.find((d) => d.id === tier) ?? DIFFICULTIES[2];

  useEffect(() => {
    const client = clientRef.current;
    return () => {
      client?.dispose();
    };
  }, []);

  useEffect(() => {
    clientRef.current?.preload(tier);
  }, [tier]);

  const gameState = useMemo(
    () => replayMoves(moves.map((m) => m.col)),
    [moves],
  );

  const gameOver = gameState.winner !== 0 || gameState.isDraw;
  const isAiTurn = !gameOver && gameState.currentPlayer !== humanSide;

  // Trigger AI move only on AI's turn
  useEffect(() => {
    if (!isAiTurn) return;
    const client = clientRef.current;
    if (!client) return;

    let cancelled = false;
    setIsThinking(true);
    setHintCol(null);
    setErrorMsg(null);

    client
      .analyze(gameState, tier, activeDifficulty.sims)
      .then((res) => {
        if (cancelled) return;
        setIsThinking(false);
        if (res.bestCol >= 0) {
          setMoves((prev) => [...prev, { col: res.bestCol, player: gameState.currentPlayer }]);
        }
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof DOMException && err.name === 'AbortError') return;
        setIsThinking(false);
        setErrorMsg(err instanceof Error ? err.message : String(err));
      });

    return () => {
      cancelled = true;
    };
  }, [gameState, isAiTurn, tier, activeDifficulty.sims, reloadNonce]);

  const handlePlayColumn = useCallback(
    (col: number) => {
      if (isAiTurn || gameOver) return;
      if (!canPlay(gameState.board, col)) return;
      setHintCol(null);
      setMoves((prev) => [...prev, { col, player: gameState.currentPlayer }]);
    },
    [gameState, isAiTurn, gameOver],
  );

  // Keyboard shortcuts 1..7
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key >= '1' && e.key <= '7') {
        const col = Number(e.key) - 1;
        if (col >= 0 && col < COLS) handlePlayColumn(col);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [handlePlayColumn]);

  const handleNewGame = useCallback((nextSide?: Player) => {
    if (nextSide !== undefined) setHumanSide(nextSide);
    setMoves([]);
    setHintCol(null);
    setIsThinking(false);
    setErrorMsg(null);
    setResetCount((c) => c + 1);
  }, []);

  const handleUndo = useCallback(() => {
    if (moves.length === 0) return;
    setHintCol(null);
    setIsThinking(false);
    setMoves((prev) => {
      const next = prev.slice();
      while (next.length > 0 && next[next.length - 1].player !== humanSide) {
        next.pop();
      }
      if (next.length > 0 && next[next.length - 1].player === humanSide) {
        next.pop();
      }
      return next;
    });
  }, [moves.length, humanSide]);

  const handleToggleHint = useCallback(async () => {
    if (hintCol !== null) {
      setHintCol(null);
      return;
    }
    const client = clientRef.current;
    if (!client || gameOver || isAiTurn) return;
    setIsHintPending(true);
    try {
      const res = await client.analyze(gameState, 'pro', 64);
      setHintCol(res.bestCol >= 0 ? res.bestCol : null);
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        setErrorMsg(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setIsHintPending(false);
    }
  }, [hintCol, gameState, gameOver, isAiTurn]);

  const tierIdx = DIFFICULTIES.findIndex((d) => d.id === tier);
  const sideIdx = humanSide === 1 ? 0 : 1;

  return (
    <div className="studio-shell">
      <header className="top-bar">
        <h1 className="brand-title">Connect Four</h1>

        <div className="toolbar-controls">
          <div className="segmented" role="group" aria-label="AI Difficulty">
            <span
              className="seg-thumb"
              style={
                {
                  '--seg-idx': tierIdx,
                  '--seg-count': DIFFICULTIES.length,
                } as React.CSSProperties
              }
              aria-hidden="true"
            />
            {DIFFICULTIES.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`seg-btn ${tier === item.id ? 'active' : ''}`}
                onClick={() => {
                  setTier(item.id);
                  setHintCol(null);
                }}
              >
                {tier === item.id && (
                  <Check size={14} strokeWidth={2.5} className="seg-check" aria-hidden="true" />
                )}
                {item.label}
              </button>
            ))}
          </div>

          <div className="segmented" role="group" aria-label="Play As">
            <span
              className="seg-thumb"
              style={
                {
                  '--seg-idx': sideIdx,
                  '--seg-count': 2,
                } as React.CSSProperties
              }
              aria-hidden="true"
            />
            <button
              type="button"
              className={`seg-btn ${humanSide === 1 ? 'active' : ''}`}
              onClick={() => humanSide !== 1 && handleNewGame(1)}
            >
              <span className="swatch-dot red" />
              Red
            </button>
            <button
              type="button"
              className={`seg-btn ${humanSide === -1 ? 'active' : ''}`}
              onClick={() => humanSide !== -1 && handleNewGame(-1)}
            >
              <span className="swatch-dot yellow" />
              Yellow
            </button>
          </div>

          <button type="button" className="btn btn-new-game" onClick={() => handleNewGame()}>
            <span className="new-game-badge" aria-hidden="true">
              <Plus size={14} strokeWidth={2.5} />
            </span>
            New Game
          </button>
        </div>
      </header>

      <main className="workspace-main">
        <BoardArena
          gameState={gameState}
          humanSide={humanSide}
          isAiTurn={isAiTurn}
          isThinking={isThinking}
          hintCol={hintCol}
          isHintPending={isHintPending}
          canUndo={moves.length > 0 && !(humanSide === -1 && moves.length === 1 && isAiTurn)}
          resetCount={resetCount}
          errorMsg={errorMsg}
          onUndo={handleUndo}
          onToggleHint={handleToggleHint}
          onRetryLoad={() => {
            setErrorMsg(null);
            setReloadNonce((n) => n + 1);
          }}
          onPlayColumn={handlePlayColumn}
        />
      </main>
    </div>
  );
}

export default App;
