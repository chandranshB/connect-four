"""Connect Four bitboard environment, tensor observation encoder, and Minimax evaluator."""

from __future__ import annotations

import numpy as np

ROWS = 6
COLS = 7
NUM_ACTIONS = COLS
STRIDE = 7  # 6 playable rows + 1 sentinel bit per column

BOTTOM_MASK = tuple(1 << (c * STRIDE) for c in range(COLS))
TOP_MASK = tuple(1 << (c * STRIDE + ROWS - 1) for c in range(COLS))
COL_MASK = tuple(((1 << ROWS) - 1) << (c * STRIDE) for c in range(COLS))
BOARD_MASK = sum(COL_MASK)

# Column evaluation order from center outward (improves alpha-beta pruning)
CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)

# Precompute bitmasks for the 69 possible 4-in-a-row windows (used by heuristic Minimax benchmark)
def _build_windows() -> tuple[int, ...]:
    windows: list[int] = []
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            w = 0
            for i in range(4):
                w |= 1 << ((c + i) * STRIDE + r)
            windows.append(w)
    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            w = 0
            for i in range(4):
                w |= 1 << (c * STRIDE + r + i)
            windows.append(w)
    # Diagonal +/+
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            w = 0
            for i in range(4):
                w |= 1 << ((c + i) * STRIDE + r + i)
            windows.append(w)
    # Diagonal +/-
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            w = 0
            for i in range(4):
                w |= 1 << ((c + i) * STRIDE + r - i)
            windows.append(w)
    return tuple(windows)


WIN_WINDOWS = _build_windows()

# Center column bitmasks for subtle positional tie-breaking in Minimax heuristic
CENTER_COL_MASK = COL_MASK[3]
NEAR_CENTER_MASK = COL_MASK[2] | COL_MASK[4]


def is_win(pos: int) -> bool:
    """Return True if bitboard `pos` contains 4-in-a-row."""
    m = pos & (pos >> 1)
    if m & (m >> 2):
        return True
    m = pos & (pos >> STRIDE)
    if m & (m >> (2 * STRIDE)):
        return True
    m = pos & (pos >> (STRIDE - 1))
    if m & (m >> (2 * (STRIDE - 1))):
        return True
    m = pos & (pos >> (STRIDE + 1))
    if m & (m >> (2 * (STRIDE + 1))):
        return True
    return False


def can_play(mask: int, col: int) -> bool:
    """Return True if column `col` has at least one empty slot."""
    return (mask & TOP_MASK[col]) == 0


def legal_actions(mask: int) -> list[int]:
    """Return list of valid column indices in ascending order."""
    return [c for c in range(COLS) if (mask & TOP_MASK[c]) == 0]


def legal_mask_array(mask: int) -> np.ndarray:
    """Return boolean array of shape (7,) marking valid columns."""
    return np.array([(mask & TOP_MASK[c]) == 0 for c in range(COLS)], dtype=bool)


def play_move(pos: int, mask: int, ply: int, col: int) -> tuple[int, int, int]:
    """Apply move in column `col` and return (next_pos, next_mask, next_ply)."""
    new_mask = mask | (mask + BOTTOM_MASK[col])
    next_pos = pos ^ mask
    return next_pos, new_mask, ply + 1


def last_mover_won(pos: int, mask: int) -> bool:
    """Return True if the player who made the previous move won."""
    return is_win(pos ^ mask)


def is_draw(ply: int) -> bool:
    """Return True if all 42 cells are filled (assuming caller checked win first)."""
    return ply >= ROWS * COLS


def bitboard_to_grid(pos: int, mask: int, ply: int) -> np.ndarray:
    """Convert bitboard to (6, 7) int8 array using +1 for Player 1 (Red), -1 for Player 2 (Yellow), 0 empty.
    Row 0 is top of the board, Row 5 is bottom of the board.
    """
    p1_mask = pos if (ply % 2 == 0) else (pos ^ mask)
    p2_mask = (pos ^ mask) if (ply % 2 == 0) else pos
    grid = np.zeros((ROWS, COLS), dtype=np.int8)
    for c in range(COLS):
        base = c * STRIDE
        for r_bot in range(ROWS):
            bit = 1 << (base + r_bot)
            row = ROWS - 1 - r_bot
            if p1_mask & bit:
                grid[row, c] = 1
            elif p2_mask & bit:
                grid[row, c] = -1
    return grid


def encode_obs(pos: int, mask: int, ply: int) -> np.ndarray:
    """Encode canonical state as float32 tensor of shape (3, 6, 7).
    Channel 0: current player's discs (1.0/0.0)
    Channel 1: opponent's discs (1.0/0.0)
    Channel 2: 1.0 everywhere if Player 1 to move (ply even), else 0.0 everywhere.
    Row 0 is top of board, Row 5 is bottom of board.
    """
    obs = np.zeros((3, ROWS, COLS), dtype=np.float32)
    opp = pos ^ mask
    for c in range(COLS):
        base = c * STRIDE
        for r_bot in range(ROWS):
            bit = 1 << (base + r_bot)
            row = ROWS - 1 - r_bot
            if pos & bit:
                obs[0, row, c] = 1.0
            elif opp & bit:
                obs[1, row, c] = 1.0
    if ply % 2 == 0:
        obs[2, :, :] = 1.0
    return obs


# Precompute per-column bit extraction masks for fast vectorized batch encoding
_COL_BITS = np.array(
    [[1 << (c * STRIDE + (ROWS - 1 - r)) for c in range(COLS)] for r in range(ROWS)],
    dtype=np.int64,
)


def encode_obs_batch(states: list[tuple[int, int, int]]) -> np.ndarray:
    """Vectorized encoding of N (pos, mask, ply) states into (N, 3, 6, 7) float32."""
    n = len(states)
    pos_arr = np.fromiter((s[0] for s in states), dtype=np.int64, count=n)[:, None, None]
    opp_arr = np.fromiter((s[0] ^ s[1] for s in states), dtype=np.int64, count=n)[:, None, None]
    ply_arr = np.fromiter((s[2] for s in states), dtype=np.int64, count=n)

    obs = np.empty((n, 3, ROWS, COLS), dtype=np.float32)
    obs[:, 0] = ((pos_arr & _COL_BITS[None, :, :]) != 0).astype(np.float32)
    obs[:, 1] = ((opp_arr & _COL_BITS[None, :, :]) != 0).astype(np.float32)
    obs[:, 2] = ((ply_arr & 1) == 0).astype(np.float32)[:, None, None]
    return obs


def flip_obs(obs: np.ndarray) -> np.ndarray:
    """Horizontally mirror observation(s) of shape (..., 3, 6, 7)."""
    return obs[..., ::-1].copy()


def immediate_win_cols(pos: int, mask: int) -> list[int]:
    """Return legal columns where current player wins immediately."""
    wins: list[int] = []
    for c in CENTER_ORDER:
        if can_play(mask, c):
            new_mask = mask | (mask + BOTTOM_MASK[c])
            move_bit = new_mask ^ mask
            if is_win(pos | move_bit):
                wins.append(c)
    return wins


class MinimaxBot:
    """Alpha-beta Minimax benchmark opponent with transposition table."""

    def __init__(self, depth: int = 4) -> None:
        self.depth = depth
        self.tt: dict[tuple[int, int, int], tuple[float, int]] = {}

    def _heuristic(self, pos: int, opp: int) -> float:
        score = 0.0
        score += 3.0 * ((pos & CENTER_COL_MASK).bit_count() - (opp & CENTER_COL_MASK).bit_count())
        score += 1.2 * ((pos & NEAR_CENTER_MASK).bit_count() - (opp & NEAR_CENTER_MASK).bit_count())
        for w in WIN_WINDOWS:
            p_has = pos & w
            o_has = opp & w
            if p_has and o_has:
                continue
            if p_has:
                cnt = p_has.bit_count()
                if cnt == 3:
                    score += 14.0
                elif cnt == 2:
                    score += 3.0
                else:
                    score += 0.5
            elif o_has:
                cnt = o_has.bit_count()
                if cnt == 3:
                    score -= 16.0
                elif cnt == 2:
                    score -= 3.2
                else:
                    score -= 0.5
        return score

    def _alphabeta(
        self, pos: int, mask: int, ply: int, depth: int, alpha: float, beta: float
    ) -> float:
        if last_mover_won(pos, mask):
            # Previous player just won; prefer faster wins / slower losses
            return -(10000.0 + depth * 100.0)
        if ply >= ROWS * COLS:
            return 0.0
        if depth == 0:
            return self._heuristic(pos, pos ^ mask)

        # Quick check: if current player can win in 1, return immediately
        for c in CENTER_ORDER:
            if can_play(mask, c):
                new_mask = mask | (mask + BOTTOM_MASK[c])
                if is_win(pos | (new_mask ^ mask)):
                    return 10000.0 + (depth - 1) * 100.0

        key = (pos, mask, depth)
        cached = self.tt.get(key)
        if cached is not None:
            val, flag = cached
            if flag == 0:
                return val
            if flag == 1 and val >= beta:
                return val
            if flag == -1 and val <= alpha:
                return val

        orig_alpha = alpha
        best = -1e9
        for c in CENTER_ORDER:
            if not can_play(mask, c):
                continue
            np_pos, np_mask, np_ply = play_move(pos, mask, ply, c)
            val = -self._alphabeta(np_pos, np_mask, np_ply, depth - 1, -beta, -alpha)
            if val > best:
                best = val
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break

        if best <= orig_alpha:
            flag = -1
        elif best >= beta:
            flag = 1
        else:
            flag = 0
        if len(self.tt) < 200_000:
            self.tt[key] = (best, flag)
        return best

    def select_action(self, pos: int, mask: int, ply: int) -> int:
        self.tt.clear()
        # Immediate win first
        wins = immediate_win_cols(pos, mask)
        if wins:
            return wins[0]
        # Must-block check
        opp_wins = immediate_win_cols(pos ^ mask, mask)
        if opp_wins:
            return opp_wins[0]

        best_col = -1
        best_val = -1e9
        alpha = -1e9
        beta = 1e9
        for c in CENTER_ORDER:
            if not can_play(mask, c):
                continue
            np_pos, np_mask, np_ply = play_move(pos, mask, ply, c)
            val = -self._alphabeta(np_pos, np_mask, np_ply, self.depth - 1, -beta, -alpha)
            if val > best_val:
                best_val = val
                best_col = c
            if val > alpha:
                alpha = val
        return best_col
