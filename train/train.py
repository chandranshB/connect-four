"""AlphaZero self-play trainer, batched tactical benchmark suite, and checkpoint saver."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from c4_env import (
    CENTER_ORDER,
    MinimaxBot,
    TOP_MASK,
    encode_obs_batch,
    immediate_win_cols,
    is_draw,
    last_mover_won,
    legal_actions,
    play_move,
)
from mcts import MCTSNode, _softmax_masked, generate_selfplay_batch, run_batched_mcts
from model import Connect4Net, count_parameters


class ReplayBuffer:
    def __init__(self, capacity: int = 160_000) -> None:
        self.capacity = capacity
        self.obs: np.ndarray | None = None
        self.pi: np.ndarray | None = None
        self.z: np.ndarray | None = None
        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, pi: np.ndarray, z: np.ndarray) -> None:
        n = obs.shape[0]
        if self.obs is None:
            self.obs = np.empty((self.capacity, *obs.shape[1:]), dtype=np.float32)
            self.pi = np.empty((self.capacity, pi.shape[1]), dtype=np.float32)
            self.z = np.empty((self.capacity,), dtype=np.float32)

        assert self.obs is not None and self.pi is not None and self.z is not None
        if n >= self.capacity:
            self.obs[:] = obs[-self.capacity :]
            self.pi[:] = pi[-self.capacity :]
            self.z[:] = z[-self.capacity :]
            self.ptr = 0
            self.size = self.capacity
            return

        end = self.ptr + n
        if end <= self.capacity:
            self.obs[self.ptr : end] = obs
            self.pi[self.ptr : end] = pi
            self.z[self.ptr : end] = z
        else:
            first = self.capacity - self.ptr
            self.obs[self.ptr :] = obs[:first]
            self.pi[self.ptr :] = pi[:first]
            self.z[self.ptr :] = z[:first]
            rem = n - first
            self.obs[:rem] = obs[first:]
            self.pi[:rem] = pi[first:]
            self.z[:rem] = z[first:]
        self.ptr = end % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.obs is not None and self.pi is not None and self.z is not None
        idx = rng.integers(0, self.size, size=batch_size)
        return self.obs[idx], self.pi[idx], self.z[idx]


@torch.inference_mode()
def play_match_batched(
    model: torch.nn.Module,
    device: torch.device,
    num_sims: int,
    opponent_depth: int,
    num_games: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """Play `num_games` concurrently against MinimaxBot(depth), batching AI GPU evals."""
    model.eval()
    rng = np.random.default_rng(seed)
    opp = MinimaxBot(depth=opponent_depth)

    games: list[list[int]] = []  # [pos, mask, ply, a_is_p1]
    for g in range(num_games):
        pos, mask, ply = 0, 0, 0
        if g >= 2:
            for _ in range(2):
                legal = legal_actions(mask)
                c = int(rng.choice(legal))
                pos, mask, ply = play_move(pos, mask, ply, c)
        games.append([pos, mask, ply, 1 if (g % 2 == 0) else 0])

    active = list(range(num_games))
    wins = 0
    losses = 0
    draws = 0

    while active:
        ai_turn_indices: list[int] = []
        next_active: list[int] = []

        for g_idx in active:
            pos, mask, ply, a_is_p1 = games[g_idx]
            if last_mover_won(pos, mask):
                last_p1 = 1 if ((ply - 1) % 2 == 0) else 0
                if last_p1 == a_is_p1:
                    wins += 1
                else:
                    losses += 1
                continue
            if is_draw(ply):
                draws += 1
                continue

            p1_turn = 1 if (ply % 2 == 0) else 0
            if p1_turn == a_is_p1:
                ai_turn_indices.append(g_idx)
            else:
                col = opp.select_action(pos, mask, ply)
                np_pos, np_mask, np_ply = play_move(pos, mask, ply, col)
                games[g_idx][0], games[g_idx][1], games[g_idx][2] = np_pos, np_mask, np_ply
            next_active.append(g_idx)

        if ai_turn_indices:
            if num_sims <= 0:
                states = [(games[i][0], games[i][1], games[i][2]) for i in ai_turn_indices]
                obs = encode_obs_batch(states)
                logits, _ = model(torch.from_numpy(obs).to(device))
                logits_np = logits.detach().float().cpu().numpy()
                for k, g_idx in enumerate(ai_turn_indices):
                    pos, mask, ply, _ = games[g_idx]
                    legal = [c for c in CENTER_ORDER if (mask & TOP_MASK[c]) == 0]
                    probs = _softmax_masked(logits_np[k], legal)
                    col = max(legal, key=lambda c: float(probs[c]))
                    np_pos, np_mask, np_ply = play_move(pos, mask, ply, col)
                    games[g_idx][0], games[g_idx][1], games[g_idx][2] = np_pos, np_mask, np_ply
            else:
                roots = [MCTSNode(games[i][0], games[i][1], games[i][2]) for i in ai_turn_indices]
                policies = run_batched_mcts(model, device, roots, num_sims=num_sims, add_noise=False)
                for k, g_idx in enumerate(ai_turn_indices):
                    pos, mask, ply, _ = games[g_idx]
                    legal = roots[k].legal_cols
                    pi = policies[k]
                    col = max(legal, key=lambda c: float(pi[c]))
                    np_pos, np_mask, np_ply = play_move(pos, mask, ply, col)
                    games[g_idx][0], games[g_idx][1], games[g_idx][2] = np_pos, np_mask, np_ply

        active = next_active

    score = (wins + 0.5 * draws) / num_games
    return {"wins": wins, "losses": losses, "draws": draws, "score": score}


@torch.inference_mode()
def evaluate_tactics(model: torch.nn.Module, device: torch.device, seed: int = 123) -> dict[str, float]:
    """Test raw 0-sim policy accuracy on immediate win-in-1 and block-in-1 board states."""
    model.eval()
    rng = np.random.default_rng(seed)
    win_states: list[tuple[int, int, int, set[int]]] = []
    block_states: list[tuple[int, int, int, set[int]]] = []

    attempts = 0
    while (len(win_states) < 120 or len(block_states) < 120) and attempts < 5000:
        attempts += 1
        pos, mask, ply = 0, 0, 0
        steps = int(rng.integers(5, 26))
        for _ in range(steps):
            if last_mover_won(pos, mask) or is_draw(ply):
                break
            legal = legal_actions(mask)
            c = int(rng.choice(legal))
            pos, mask, ply = play_move(pos, mask, ply, c)
        if last_mover_won(pos, mask) or is_draw(ply):
            continue

        cur_wins = set(immediate_win_cols(pos, mask))
        opp_wins = set(immediate_win_cols(pos ^ mask, mask))
        if cur_wins and len(win_states) < 120:
            win_states.append((pos, mask, ply, cur_wins))
        elif not cur_wins and opp_wins and len(block_states) < 120:
            block_states.append((pos, mask, ply, opp_wins))

    def _batch_acc(items: list[tuple[int, int, int, set[int]]]) -> float:
        if not items:
            return 0.0
        obs = encode_obs_batch([(p, m, pl) for p, m, pl, _ in items])
        logits, _ = model(torch.from_numpy(obs).to(device))
        logits_np = logits.detach().float().cpu().numpy()
        hits = 0
        for i, (_, mask, _, ans) in enumerate(items):
            legal = [c for c in CENTER_ORDER if (mask & TOP_MASK[c]) == 0]
            best = max(legal, key=lambda c: float(logits_np[i, c]))
            if best in ans:
                hits += 1
        return hits / len(items)

    return {
        "raw_win1_acc": _batch_acc(win_states),
        "raw_block1_acc": _batch_acc(block_states),
    }


def save_checkpoint(path: Path, model: Connect4Net, iteration: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "channels": model.channels,
            "num_blocks": model.num_blocks,
            "state_dict": model.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Connect Four AlphaZero AI")
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--games-per-iter", type=int, default=96)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--steps-per-iter", type=int, default=140)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using device: {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        flush=True,
    )

    model = Connect4Net(channels=args.channels, num_blocks=args.blocks).to(device)
    print(f"Model parameters: {count_parameters(model):,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iterations, eta_min=2e-4
    )

    buffer = ReplayBuffer(capacity=160_000)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_pro_score = -1.0
    total_games = 0
    total_samples = 0

    early_iter = min(args.iterations, max(2, args.iterations // 6))
    mid_iter = min(args.iterations, max(early_iter + 1, args.iterations // 2))

    start_all = time.perf_counter()

    for it in range(1, args.iterations + 1):
        t0 = time.perf_counter()
        iter_sims = args.sims if it <= args.iterations // 2 else int(args.sims * 1.25)

        obs_aug, pi_aug, z_aug, sp_stats = generate_selfplay_batch(
            model=model,
            device=device,
            num_games=args.games_per_iter,
            num_sims=iter_sims,
            temp_threshold_ply=10,
            c_puct=1.5,
            rng=rng,
        )
        t_sp = time.perf_counter() - t0

        buffer.add(obs_aug, pi_aug, z_aug)
        total_games += args.games_per_iter
        total_samples += len(z_aug)

        t1 = time.perf_counter()
        model.train()
        p_losses = []
        v_losses = []
        for _ in range(args.steps_per_iter):
            b_obs, b_pi, b_z = buffer.sample(args.batch_size, rng)
            x_t = torch.from_numpy(b_obs).to(device=device, non_blocking=True)
            pi_t = torch.from_numpy(b_pi).to(device=device, non_blocking=True)
            z_t = torch.from_numpy(b_z).to(device=device, non_blocking=True)

            logits, val = model(x_t)
            log_probs = F.log_softmax(logits, dim=-1)
            p_loss = -(pi_t * log_probs).sum(dim=-1).mean()
            v_loss = F.mse_loss(val, z_t)
            loss = p_loss + 1.25 * v_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            p_losses.append(float(p_loss.item()))
            v_losses.append(float(v_loss.item()))

        scheduler.step()
        t_tr = time.perf_counter() - t1

        rec: dict = {
            "iteration": it,
            "total_games": total_games,
            "total_samples": total_samples,
            "buffer_size": buffer.size,
            "policy_loss": round(float(np.mean(p_losses)), 4),
            "value_loss": round(float(np.mean(v_losses)), 4),
            "avg_game_len": round(sp_stats["avg_len"], 2),
            "p1_win_pct": round(100.0 * sp_stats["p1_wins"] / args.games_per_iter, 1),
            "lr": round(float(scheduler.get_last_lr()[0]), 6),
            "selfplay_s": round(t_sp, 2),
            "train_s": round(t_tr, 2),
        }

        should_eval = (it in (1, early_iter, mid_iter, args.iterations)) or (it % 6 == 0)
        if should_eval:
            tactics = evaluate_tactics(model, device)
            rec["raw_win1_acc"] = round(100.0 * tactics["raw_win1_acc"], 1)
            rec["raw_block1_acc"] = round(100.0 * tactics["raw_block1_acc"], 1)

            vs_mm2_raw = play_match_batched(model, device, num_sims=0, opponent_depth=2, num_games=12, seed=100 + it)
            vs_mm2 = play_match_batched(model, device, num_sims=64, opponent_depth=2, num_games=12, seed=200 + it)
            vs_mm4 = play_match_batched(model, device, num_sims=64, opponent_depth=4, num_games=12, seed=300 + it)
            rec["raw_vs_mm2"] = round(100.0 * vs_mm2_raw["score"], 1)
            rec["mcts64_vs_mm2"] = round(100.0 * vs_mm2["score"], 1)
            rec["mcts64_vs_mm4"] = round(100.0 * vs_mm4["score"], 1)

            if it >= mid_iter and vs_mm4["score"] >= best_pro_score:
                best_pro_score = vs_mm4["score"]
                save_checkpoint(args.out_dir / "pro_best.pt", model, it, rec)

        history.append(rec)
        eval_str = ""
        if "raw_win1_acc" in rec:
            eval_str = (
                f" | win1={rec['raw_win1_acc']}% blk1={rec['raw_block1_acc']}% "
                f"raw/mm2={rec['raw_vs_mm2']}% mcts64/mm2={rec['mcts64_vs_mm2']}% "
                f"mcts64/mm4={rec['mcts64_vs_mm4']}%"
            )
        print(
            f"[Iter {it:02d}/{args.iterations}] games={total_games} len={rec['avg_game_len']} "
            f"p_loss={rec['policy_loss']:.3f} v_loss={rec['value_loss']:.3f} "
            f"({t_sp:.1f}s sp + {t_tr:.1f}s tr){eval_str}",
            flush=True,
        )

        if it == early_iter:
            save_checkpoint(args.out_dir / "early.pt", model, it, rec)
        if it == mid_iter:
            save_checkpoint(args.out_dir / "mid.pt", model, it, rec)
        if it == args.iterations:
            save_checkpoint(args.out_dir / "pro_last.pt", model, it, rec)

    pro_last_ckpt = torch.load(args.out_dir / "pro_last.pt", map_location="cpu", weights_only=False)
    pro_best_path = args.out_dir / "pro_best.pt"
    if pro_best_path.exists():
        pro_best_ckpt = torch.load(pro_best_path, map_location="cpu", weights_only=False)
        last_score = pro_last_ckpt["metrics"].get("mcts64_vs_mm4", 0.0)
        best_score = pro_best_ckpt["metrics"].get("mcts64_vs_mm4", 0.0)
        final_pro_src = pro_best_path if best_score > last_score + 5.0 else (args.out_dir / "pro_last.pt")
    else:
        final_pro_src = args.out_dir / "pro_last.pt"

    final_pro_ckpt = torch.load(final_pro_src, map_location="cpu", weights_only=False)
    torch.save(final_pro_ckpt, args.out_dir / "pro.pt")

    summary_tiers = {}
    for tier in ("early", "mid", "pro"):
        ckpt = torch.load(args.out_dir / f"{tier}.pt", map_location="cpu", weights_only=False)
        m = Connect4Net(channels=ckpt["channels"], num_blocks=ckpt["num_blocks"]).to(device)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        tactics = evaluate_tactics(m, device)
        raw_res = play_match_batched(m, device, num_sims=0, opponent_depth=2, num_games=20, seed=500)
        mcts_res_2 = play_match_batched(m, device, num_sims=64, opponent_depth=2, num_games=20, seed=600)
        mcts_res_4 = play_match_batched(m, device, num_sims=96, opponent_depth=4, num_games=20, seed=700)
        summary_tiers[tier] = {
            "iteration": ckpt["iteration"],
            "raw_win1_acc": round(100.0 * tactics["raw_win1_acc"], 1),
            "raw_block1_acc": round(100.0 * tactics["raw_block1_acc"], 1),
            "raw_vs_mm2_pct": round(100.0 * raw_res["score"], 1),
            "mcts64_vs_mm2_pct": round(100.0 * mcts_res_2["score"], 1),
            "mcts96_vs_mm4_pct": round(100.0 * mcts_res_4["score"], 1),
        }

    elapsed = time.perf_counter() - start_all
    report = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "params": count_parameters(model),
        "elapsed_seconds": round(elapsed, 1),
        "total_selfplay_games": total_games,
        "total_augmented_samples": total_samples,
        "tiers": summary_tiers,
        "history": history,
    }
    (args.out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== Final Checkpoint Summary ===", flush=True)
    print(json.dumps(summary_tiers, indent=2), flush=True)
    print(f"Total elapsed: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
