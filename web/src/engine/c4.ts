export const ROWS = 6;
export const COLS = 7;
export const NUM_CELLS = ROWS * COLS;
export const CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6] as const;

export type Player = 1 | -1; // 1 = Red (first mover), -1 = Yellow (second mover)
export type Cell = 0 | Player;

export interface Coord {
  row: number;
  col: number;
}

export interface GameState {
  board: Int8Array; // length 42, row-major (row 0 = top, row 5 = bottom)
  ply: number;
  currentPlayer: Player;
  winner: Player | 0;
  isDraw: boolean;
  winningLine: Coord[];
  lastMove: Coord | null;
}

export function createInitialState(): GameState {
  return {
    board: new Int8Array(NUM_CELLS),
    ply: 0,
    currentPlayer: 1,
    winner: 0,
    isDraw: false,
    winningLine: [],
    lastMove: null,
  };
}

export function cloneState(s: GameState): GameState {
  return {
    board: new Int8Array(s.board),
    ply: s.ply,
    currentPlayer: s.currentPlayer,
    winner: s.winner,
    isDraw: s.isDraw,
    winningLine: s.winningLine.slice(),
    lastMove: s.lastMove ? { ...s.lastMove } : null,
  };
}

export function getDropRow(board: Int8Array, col: number): number {
  if (col < 0 || col >= COLS) return -1;
  for (let r = ROWS - 1; r >= 0; r--) {
    if (board[r * COLS + col] === 0) return r;
  }
  return -1;
}

export function canPlay(board: Int8Array, col: number): boolean {
  return col >= 0 && col < COLS && board[col] === 0;
}

export function getLegalCols(board: Int8Array): number[] {
  const cols: number[] = [];
  for (const c of CENTER_ORDER) {
    if (board[c] === 0) cols.push(c);
  }
  return cols;
}

export function findWinningLine(board: Int8Array, row: number, col: number, player: Player): Coord[] {
  const dirs: [number, number][] = [
    [0, 1],
    [1, 0],
    [1, 1],
    [1, -1],
  ];
  const allWinCoords: Coord[] = [];
  const seen = new Set<number>();

  for (const [dr, dc] of dirs) {
    const line: Coord[] = [{ row, col }];
    for (const sign of [-1, 1]) {
      let r = row + sign * dr;
      let c = col + sign * dc;
      while (r >= 0 && r < ROWS && c >= 0 && c < COLS && board[r * COLS + c] === player) {
        line.push({ row: r, col: c });
        r += sign * dr;
        c += sign * dc;
      }
    }
    if (line.length >= 4) {
      for (const cell of line) {
        const key = cell.row * COLS + cell.col;
        if (!seen.has(key)) {
          seen.add(key);
          allWinCoords.push(cell);
        }
      }
    }
  }
  return allWinCoords;
}

export function applyMove(state: GameState, col: number): GameState {
  if (state.winner !== 0 || state.isDraw) {
    throw new Error('Cannot play on a finished game');
  }
  const row = getDropRow(state.board, col);
  if (row < 0) {
    throw new Error(`Illegal column ${col}`);
  }
  const nextBoard = new Int8Array(state.board);
  const mover = state.currentPlayer;
  nextBoard[row * COLS + col] = mover;

  const winningLine = findWinningLine(nextBoard, row, col, mover);
  const nextPly = state.ply + 1;
  const winner: Player | 0 = winningLine.length >= 4 ? mover : 0;
  const isDraw = winner === 0 && nextPly >= NUM_CELLS;

  return {
    board: nextBoard,
    ply: nextPly,
    currentPlayer: (mover === 1 ? -1 : 1) as Player,
    winner,
    isDraw,
    winningLine,
    lastMove: { row, col },
  };
}

export function replayMoves(moves: number[]): GameState {
  let state = createInitialState();
  for (const col of moves) {
    state = applyMove(state, col);
  }
  return state;
}

/**
 * Encode canonical (3, 6, 7) CHW float32 tensor matching Python `c4_env.encode_obs`.
 * Channel 0: currentPlayer discs (1.0/0.0)
 * Channel 1: opponent discs (1.0/0.0)
 * Channel 2: 1.0 everywhere if Player 1 (Red, ply even) to move, else 0.0
 */
export function encodeObservation(state: GameState): Float32Array {
  const out = new Float32Array(3 * NUM_CELLS);
  const cur = state.currentPlayer;
  const opp: Player = cur === 1 ? -1 : 1;
  const isP1Turn = state.ply % 2 === 0;

  for (let i = 0; i < NUM_CELLS; i++) {
    const v = state.board[i];
    if (v === cur) {
      out[i] = 1.0;
    } else if (v === opp) {
      out[NUM_CELLS + i] = 1.0;
    }
    if (isP1Turn) {
      out[2 * NUM_CELLS + i] = 1.0;
    }
  }
  return out;
}

export function immediateWinningCols(state: GameState, player: Player): number[] {
  const wins: number[] = [];
  for (const c of CENTER_ORDER) {
    const r = getDropRow(state.board, c);
    if (r < 0) continue;
    state.board[r * COLS + c] = player;
    const line = findWinningLine(state.board, r, c, player);
    state.board[r * COLS + c] = 0;
    if (line.length >= 4) wins.push(c);
  }
  return wins;
}
