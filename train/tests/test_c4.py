"""Unit tests for Connect Four bitboard rules, symmetry, observation encoding, and MCTS."""

from __future__ import annotations

import numpy as np
import torch

from c4_env import (
    COLS,
    ROWS,
    MinimaxBot,
    bitboard_to_grid,
    can_play,
    encode_obs,
    encode_obs_batch,
    flip_obs,
    immediate_win_cols,
    is_draw,
    is_win,
    last_mover_won,
    legal_actions,
    play_move,
)
from mcts import MCTSNode, run_batched_mcts
from model import Connect4Net


def _naive_check_win(grid: np.ndarray, player: int) -> bool:
    for r in range(ROWS):
        for c in range(COLS):
            if c + 3 < COLS and all(grid[r, c + i] == player for i in range(4)):
                return True
            if r + 3 < ROWS and all(grid[r + i, c] == player for i in range(4)):
                return True
            if r + 3 < ROWS and c + 3 < COLS and all(grid[r + i, c + i] == player for i in range(4)):
                return True
            if r - 3 >= 0 and c + 3 < COLS and all(grid[r - i, c + i] == player for i in range(4)):
                return True
    return False


def test_bitboard_matches_naive_grid_across_random_games() -> None:
    rng = np.random.default_rng(12345)
    for _ in range(300):
        grid = np.zeros((ROWS, COLS), dtype=np.int8)
        heights = [0] * COLS
        pos, mask, ply = 0, 0, 0

        while True:
            legal = [c for c in range(COLS) if heights[c] < ROWS]
            assert legal == legal_actions(mask)
            for c in range(COLS):
                assert can_play(mask, c) == (heights[c] < ROWS)

            bb_grid = bitboard_to_grid(pos, mask, ply)
            assert np.array_equal(bb_grid, grid)

            if not legal:
                assert is_draw(ply)
                break

            c = int(rng.choice(legal))
            player = 1 if (ply % 2 == 0) else -1
            row = ROWS - 1 - heights[c]
            grid[row, c] = player
            heights[c] += 1
            pos, mask, ply = play_move(pos, mask, ply, c)

            naive_win = _naive_check_win(grid, player)
            bb_win = last_mover_won(pos, mask)
            assert bb_win == naive_win
            if bb_win:
                break


def test_encode_obs_and_batch_and_flip_symmetry() -> None:
    pos, mask, ply = 0, 0, 0
    for c in (3, 2, 3, 2, 4):
        pos, mask, ply = play_move(pos, mask, ply, c)

    single = encode_obs(pos, mask, ply)
    batch = encode_obs_batch([(pos, mask, ply)])
    assert batch.shape == (1, 3, 6, 7)
    assert np.allclose(single, batch[0])

    # Current player at ply=5 is Player 2 (-1), who played col 2 twice -> bottom two cells of col 2
    assert single[0, 5, 2] == 1.0 and single[0, 4, 2] == 1.0
    # Opponent is Player 1 (+1), who played col 3 twice and col 4 once
    assert single[1, 5, 3] == 1.0 and single[1, 4, 3] == 1.0 and single[1, 5, 4] == 1.0
    # Turn plane is 0.0 when Player 2 to move
    assert np.all(single[2] == 0.0)

    # Mirror symmetry test: playing mirrored moves gives mirrored observation
    pos_m, mask_m, ply_m = 0, 0, 0
    for c in (6 - x for x in (3, 2, 3, 2, 4)):
        pos_m, mask_m, ply_m = play_move(pos_m, mask_m, ply_m, c)
    assert np.allclose(flip_obs(single), encode_obs(pos_m, mask_m, ply_m))


def test_sentinel_prevents_column_wrap_wins() -> None:
    # Fill top of col 0 and bottom of col 1 — must NOT trigger false 4-in-a-row
    pos = (1 << 4) | (1 << 5) | (1 << 7) | (1 << 8)
    assert not is_win(pos)

    # Fill row 0 across cols 0..3 -> valid horizontal win
    pos_h = (1 << 0) | (1 << 7) | (1 << 14) | (1 << 21)
    assert is_win(pos_h)


def test_immediate_win_and_mcts_finds_win_in_1() -> None:
    pos, mask, ply = 0, 0, 0
    # P1 plays col 0 x3, P2 plays col 6 x3 -> P1 has immediate win in col 0
    for c in (0, 6, 0, 6, 0, 6):
        pos, mask, ply = play_move(pos, mask, ply, c)

    assert immediate_win_cols(pos, mask) == [0]

    model = Connect4Net(channels=16, num_blocks=1)
    model.eval()
    root = MCTSNode(pos=pos, mask=mask, ply=ply)
    policies = run_batched_mcts(model, torch.device("cpu"), [root], num_sims=32, add_noise=False)
    assert int(np.argmax(policies[0])) == 0


def test_minimax_takes_win_and_blocks() -> None:
    bot = MinimaxBot(depth=2)
    pos, mask, ply = 0, 0, 0
    for c in (3, 0, 3, 1, 3):  # P1 has 3 in col 3; P2 to move MUST block col 3
        pos, mask, ply = play_move(pos, mask, ply, c)
    assert bot.select_action(pos, mask, ply) == 3
