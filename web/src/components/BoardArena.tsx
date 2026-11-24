import { useState } from 'react';
import { Lightbulb, RotateCcw } from 'lucide-react';
import { COLS, ROWS, getDropRow, type GameState, type Player } from '../engine/c4';

interface BoardArenaProps {
  gameState: GameState;
  humanSide: Player;
  isAiTurn: boolean;
  isThinking: boolean;
  hintCol: number | null;
  isHintPending: boolean;
  canUndo: boolean;
  resetCount: number;
  errorMsg: string | null;
  onUndo: () => void;
  onToggleHint: () => void;
  onRetryLoad: () => void;
  onPlayColumn: (col: number) => void;
}

export function BoardArena({
  gameState,
  humanSide,
  isAiTurn,
  isThinking,
  hintCol,
  isHintPending,
  canUndo,
  resetCount,
  errorMsg,
  onUndo,
  onToggleHint,
  onRetryLoad,
  onPlayColumn,
}: BoardArenaProps) {
  const [hoveredCol, setHoveredCol] = useState<number | null>(null);

  const gameOver = gameState.winner !== 0 || gameState.isDraw;
  const hasWinner = gameState.winner !== 0;
  const canInteract = !gameOver && !isAiTurn;

  const winningIndexMap = new Map<number, number>();
  gameState.winningLine.forEach((c, idx) => {
    winningIndexMap.set(c.row * COLS + c.col, idx);
  });

  let headline = 'Your Turn';
  let badgeClass = humanSide === 1 ? 'red' : 'yellow';
  if (gameState.winner === 1) {
    headline = humanSide === 1 ? 'You Win!' : 'AI Wins';
    badgeClass = 'red';
  } else if (gameState.winner === -1) {
    headline = humanSide === -1 ? 'You Win!' : 'AI Wins';
    badgeClass = 'yellow';
  } else if (gameState.isDraw) {
    headline = 'Draw';
    badgeClass = 'neutral';
  } else if (isThinking || isAiTurn) {
    headline = 'AI Thinking…';
    badgeClass = gameState.currentPlayer === 1 ? 'red' : 'yellow';
  }

  const hintDropRow = hintCol !== null ? getDropRow(gameState.board, hintCol) : -1;
  const hoveredDropRow = canInteract && hoveredCol !== null ? getDropRow(gameState.board, hoveredCol) : -1;

  return (
    <section className="arena-column" aria-label="Connect Four Board">
      {errorMsg && (
        <div className="error-banner" role="alert">
          <span>Model error: {errorMsg}</span>
          <button type="button" className="btn" onClick={onRetryLoad}>
            Retry
          </button>
        </div>
      )}

      <div className="arena-paper-sheet">
        <div className="status-banner">
          <div className="status-headline-wrap">
            <span
              className={`turn-badge ${badgeClass} ${isThinking && !gameOver ? 'thinking' : ''}`}
              aria-hidden="true"
            />
            <h2 key={headline} className="status-headline" aria-live="polite">
              {headline}
            </h2>
          </div>

          <div className="board-actions">
            <button
              type="button"
              className={`btn btn-compact ${hintCol !== null ? 'btn-active-toggle' : 'btn-tonal'}`}
              disabled={gameOver || isAiTurn || isHintPending}
              onClick={onToggleHint}
            >
              <Lightbulb size={15} />
              {isHintPending ? '…' : hintCol !== null ? `Col ${hintCol + 1}` : 'Hint'}
            </button>

            <button
              type="button"
              className="btn btn-compact"
              disabled={!canUndo}
              onClick={onUndo}
            >
              <RotateCcw size={15} />
              Undo
            </button>
          </div>
        </div>

        <div className="board-stack">
          {/* Hover preview drop rail */}
          <div className="drop-chute" aria-hidden="true">
            {Array.from({ length: COLS }, (_, col) => {
              const legal = getDropRow(gameState.board, col) >= 0;
              const active = canInteract && legal && hoveredCol === col;
              const discColor = gameState.currentPlayer === 1 ? 'red' : 'yellow';
              return (
                <div key={col} className="chute-slot">
                  <div className={`preview-disc ${discColor} ${active ? 'visible' : ''}`} />
                </div>
              );
            })}
          </div>

          {/* Matte 7x6 Slate Board */}
          <div key={`board-${resetCount}`} className="board-frame">
            <div
              className="board-columns-grid"
              onMouseLeave={() => setHoveredCol(null)}
            >
              {Array.from({ length: COLS }, (_, col) => {
                const dropRow = getDropRow(gameState.board, col);
                const colPlayable = canInteract && dropRow >= 0;
                return (
                  <button
                    key={col}
                    type="button"
                    className={`board-col-btn ${hoveredCol === col ? 'hovered' : ''}`}
                    disabled={!colPlayable}
                    aria-label={`Column ${col + 1}${dropRow < 0 ? ' (full)' : ''}`}
                    onMouseEnter={() => setHoveredCol(col)}
                    onFocus={() => setHoveredCol(col)}
                    onBlur={() => setHoveredCol((prev) => (prev === col ? null : prev))}
                    onClick={() => colPlayable && onPlayColumn(col)}
                  >
                    {Array.from({ length: ROWS }, (_, row) => {
                      const cellIdx = row * COLS + col;
                      const cellVal = gameState.board[cellIdx];
                      const isLast =
                        gameState.lastMove?.row === row && gameState.lastMove?.col === col;
                      const winOrder = winningIndexMap.get(cellIdx);
                      const isWin = winOrder !== undefined;
                      const isHintTarget =
                        hintCol === col && hintDropRow === row && cellVal === 0 && !gameOver;
                      const isHoverLanding =
                        hoveredCol === col && hoveredDropRow === row && cellVal === 0 && !isHintTarget;

                      return (
                        <div
                          key={row}
                          className={[
                            'board-cell',
                            isLast ? 'just-landed' : '',
                            isHintTarget ? 'hint-target' : '',
                            isHoverLanding ? 'landing-preview' : '',
                          ]
                            .filter(Boolean)
                            .join(' ')}
                        >
                          {cellVal !== 0 ? (
                            <div
                              key={isLast ? `drop-${gameState.ply}-${row}-${col}` : `disc-${row}-${col}`}
                              className={[
                                'disc',
                                cellVal === 1 ? 'red' : 'yellow',
                                isLast ? 'just-dropped last-move-marker' : '',
                                isWin ? 'win-cell' : hasWinner ? 'dimmed-disc' : '',
                              ]
                                .filter(Boolean)
                                .join(' ')}
                              style={
                                {
                                  ...(isLast
                                    ? {
                                        '--drop-offset': `${-(row + 1.35) * 116}%`,
                                        '--drop-ms': `${150 + row * 32}ms`,
                                      }
                                    : {}),
                                  ...(isWin ? { '--win-idx': winOrder } : {}),
                                } as React.CSSProperties
                              }
                            />
                          ) : isHintTarget ? (
                            <span className="hint-dot" aria-label="Recommended move" />
                          ) : null}
                        </div>
                      );
                    })}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
