import { describe, expect, it } from 'vitest';
import {
  applyMove,
  createInitialState,
  encodeObservation,
  getDropRow,
  immediateWinningCols,
  replayMoves,
} from './c4';
import { runMctsSearch, softmaxMasked, type PolicyValueEvaluator } from './mcts';

describe('Connect Four Engine', () => {
  it('drops discs to bottom-most empty row and detects vertical win', () => {
    const s = replayMoves([0, 1, 0, 1, 0, 1, 0]);
    expect(s.winner).toBe(1);
    expect(s.winningLine.length).toBe(4);
    expect(s.winningLine.map((c) => c.row).sort()).toEqual([2, 3, 4, 5]);
  });

  it('detects diagonal win and prevents moves after game over', () => {
    // P1 builds diagonal (5,0), (4,1), (3,2), (2,3)
    const s = replayMoves([
      0, // P1 (5,0)
      1, // P2 (5,1)
      1, // P1 (4,1)
      2, // P2 (5,2)
      2, // P1 (4,2)
      3, // P2 (5,3)
      2, // P1 (3,2)
      3, // P2 (4,3)
      4, // P1 (5,4)
      3, // P2 (3,3)
      3, // P1 (2,3) -> diagonal win!
    ]);
    expect(s.winner).toBe(1);
    expect(s.winningLine.length).toBe(4);
    expect(() => applyMove(s, 4)).toThrow();
  });

  it('encodes canonical 3x6x7 observation matching Python c4_env', () => {
    const s = replayMoves([3, 2, 3, 2, 4]);
    expect(s.ply).toBe(5);
    expect(s.currentPlayer).toBe(-1); // Yellow to move
    const obs = encodeObservation(s);
    expect(obs.length).toBe(126);

    // Channel 0 (current player = Yellow): (5,2) and (4,2)
    expect(obs[5 * 7 + 2]).toBe(1.0);
    expect(obs[4 * 7 + 2]).toBe(1.0);
    expect(obs[5 * 7 + 3]).toBe(0.0);

    // Channel 1 (opponent = Red): (5,3), (4,3), (5,4)
    expect(obs[42 + 5 * 7 + 3]).toBe(1.0);
    expect(obs[42 + 4 * 7 + 3]).toBe(1.0);
    expect(obs[42 + 5 * 7 + 4]).toBe(1.0);

    // Channel 2 (turn plane): 0.0 when Player 2 to move
    expect(obs[84]).toBe(0.0);
    expect(obs[125]).toBe(0.0);
  });

  it('finds immediate winning columns and MCTS picks win-in-1', async () => {
    const s = replayMoves([0, 6, 0, 6, 0, 6]);
    expect(immediateWinningCols(s, 1)).toEqual([0]);
    expect(getDropRow(s.board, 0)).toBe(2);

    const uniformEvaluator: PolicyValueEvaluator = {
      async evaluate() {
        return {
          logits: new Float32Array(7),
          value: 0,
        };
      },
    };
    const res = await runMctsSearch(s, uniformEvaluator, 40);
    expect(res.bestCol).toBe(0);
    expect(res.policyProbs[0]).toBeGreaterThan(0.4);
  });

  it('masks illegal columns cleanly in softmaxMasked', () => {
    let s = createInitialState();
    for (let i = 0; i < 6; i++) {
      s = applyMove(s, 0);
    }
    const probs = softmaxMasked(new Float32Array([10, 1, 1, 1, 1, 1, 1]), [1, 2, 3, 4, 5, 6]);
    expect(probs[0]).toBe(0);
    const sum = probs.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);
  });
});
