"""Stage-2 self-play policy sharpening & tactical polish starting from pro.pt."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from c4_env import immediate_win_cols, is_draw, last_mover_won, legal_actions, play_move
from mcts import MCTSNode, generate_selfplay_batch, run_batched_mcts
from model import Connect4Net, count_parameters
from train import ReplayBuffer, evaluate_tactics, play_match_batched, save_checkpoint


def generate_sharpened_selfplay(
    model: torch.nn.Module,
    device: torch.device,
    num_games: int = 96,
    num_sims: int = 112,
    visit_power: float = 1.75,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Self-play with sharper openings, reduced root noise, and sharpened MCTS visit targets."""
    if rng is None:
        rng = np.random.default_rng()

    model.eval()
    roots = [MCTSNode(0, 0, 0) for _ in range(num_games)]
    trajectories: list[list[tuple[tuple[int, int, int], np.ndarray, int]]] = [
        [] for _ in range(num_games)
    ]
    active_indices = list(range(num_games))

    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_plies = 0

    while active_indices:
        active_roots = [roots[i] for i in active_indices]
        # Run MCTS with lower noise & tighter exploration
        raw_policies = run_batched_mcts(
            model=model,
            device=device,
            roots=active_roots,
            num_sims=num_sims,
            c_puct=1.25,
            add_noise=True,
            dirichlet_alpha=0.6,
            noise_eps=0.12,
            rng=rng,
        )

        next_active: list[int] = []
        for idx, pi in zip(active_indices, raw_policies):
            node = roots[idx]
            player_sign = 1 if (node.ply % 2 == 0) else -1

            # Sharpen training target: exact win-in-1 if present, else power-sharpened MCTS visits
            wins = immediate_win_cols(node.pos, node.mask)
            if wins:
                target_pi = np.zeros(7, dtype=np.float32)
                for w in wins:
                    target_pi[w] = 1.0 / len(wins)
                action = wins[0]
            else:
                opp_wins = immediate_win_cols(node.pos ^ node.mask, node.mask)
                if len(opp_wins) == 1:
                    target_pi = np.zeros(7, dtype=np.float32)
                    target_pi[opp_wins[0]] = 1.0
                    action = opp_wins[0]
                else:
                    sharp = np.power(np.clip(pi, 0.0, 1.0), visit_power).astype(np.float32)
                    s = float(sharp.sum())
                    target_pi = sharp / s if s > 0 else pi.copy()
                    if node.ply < 6:
                        probs = pi.astype(np.float64)
                        probs /= probs.sum()
                        action = int(rng.choice(7, p=probs))
                    else:
                        max_v = float(np.max(target_pi))
                        best = [c for c in node.legal_cols if abs(float(target_pi[c]) - max_v) <= 1e-8]
                        action = int(best[int(rng.integers(0, len(best)))])

            trajectories[idx].append(((node.pos, node.mask, node.ply), target_pi, player_sign))

            child = node.children.get(action)
            if child is None:
                np_pos, np_mask, np_ply = play_move(node.pos, node.mask, node.ply, action)
                if last_mover_won(np_pos, np_mask):
                    child = MCTSNode(np_pos, np_mask, np_ply, is_terminal=True, terminal_value=-1.0)
                elif is_draw(np_ply):
                    child = MCTSNode(np_pos, np_mask, np_ply, is_terminal=True, terminal_value=0.0)
                else:
                    child = MCTSNode(np_pos, np_mask, np_ply)

            if child.is_terminal:
                total_plies += child.ply
                if child.terminal_value == 0.0:
                    draws += 1
                    outcome_p1 = 0.0
                else:
                    outcome_p1 = float(player_sign)
                    if player_sign == 1:
                        p1_wins += 1
                    else:
                        p2_wins += 1
                roots[idx] = child
                trajectories[idx].append(((0, 0, -1), np.empty(0, dtype=np.float32), int(outcome_p1)))
            else:
                roots[idx] = child
                next_active.append(idx)

        active_indices = next_active

    from c4_env import encode_obs_batch

    raw_states: list[tuple[int, int, int]] = []
    raw_pis: list[np.ndarray] = []
    raw_zs: list[float] = []
    for traj in trajectories:
        outcome_p1 = float(traj[-1][2])
        for st, pi_target, p_sign in traj[:-1]:
            raw_states.append(st)
            raw_pis.append(pi_target)
            raw_zs.append(outcome_p1 * float(p_sign))

    obs = encode_obs_batch(raw_states)
    pis = np.stack(raw_pis, axis=0).astype(np.float32)
    zs = np.array(raw_zs, dtype=np.float32)

    obs_aug = np.concatenate([obs, obs[:, :, :, ::-1].copy()], axis=0)
    pis_aug = np.concatenate([pis, pis[:, ::-1].copy()], axis=0)
    zs_aug = np.concatenate([zs, zs], axis=0)

    return obs_aug, pis_aug, zs_aug, {
        "p1_wins": float(p1_wins),
        "p2_wins": float(p2_wins),
        "draws": float(draws),
        "avg_len": float(total_plies) / max(1, num_games),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 tactical polish for Connect4Net pro.pt")
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--games-per-iter", type=int, default=96)
    parser.add_argument("--sims", type=int, default=96)
    parser.add_argument("--steps-per-iter", type=int, default=240)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pro_path = args.ckpt_dir / "pro.pt"
    base_ckpt = torch.load(pro_path, map_location="cpu", weights_only=False)
    base_iter = int(base_ckpt["iteration"])
    model = Connect4Net(channels=base_ckpt["channels"], num_blocks=base_ckpt["num_blocks"]).to(device)
    model.load_state_dict(base_ckpt["state_dict"])

    report_path = args.ckpt_dir / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations, eta_min=1.5e-4)
    buffer = ReplayBuffer(capacity=55_000)

    best_score = -1.0
    total_games = int(report["total_selfplay_games"])
    total_samples = int(report["total_augmented_samples"])
    t_start = time.perf_counter()

    for it in range(1, args.iterations + 1):
        t0 = time.perf_counter()
        obs_aug, pi_aug, z_aug, sp_stats = generate_sharpened_selfplay(
            model, device, num_games=args.games_per_iter, num_sims=args.sims, rng=rng
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
            b_obs, b_pi, b_z = buffer.sample(512, rng)
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

        global_it = base_iter + it
        rec: dict = {
            "iteration": global_it,
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

        eval_str = ""
        if it in (1, 4, 8, 12, args.iterations):
            tactics = evaluate_tactics(model, device)
            vs_mm2_raw = play_match_batched(model, device, num_sims=0, opponent_depth=2, num_games=16, seed=1000 + it)
            vs_mm2 = play_match_batched(model, device, num_sims=64, opponent_depth=2, num_games=16, seed=1100 + it)
            vs_mm4 = play_match_batched(model, device, num_sims=96, opponent_depth=4, num_games=16, seed=1200 + it)
            rec["raw_win1_acc"] = round(100.0 * tactics["raw_win1_acc"], 1)
            rec["raw_block1_acc"] = round(100.0 * tactics["raw_block1_acc"], 1)
            rec["raw_vs_mm2"] = round(100.0 * vs_mm2_raw["score"], 1)
            rec["mcts64_vs_mm2"] = round(100.0 * vs_mm2["score"], 1)
            rec["mcts96_vs_mm4"] = round(100.0 * vs_mm4["score"], 1)
            eval_str = (
                f" | win1={rec['raw_win1_acc']}% blk1={rec['raw_block1_acc']}% "
                f"raw/mm2={rec['raw_vs_mm2']}% mcts64/mm2={rec['mcts64_vs_mm2']}% "
                f"mcts96/mm4={rec['mcts96_vs_mm4']}%"
            )
            combined = 0.4 * vs_mm2["score"] + 0.6 * vs_mm4["score"]
            if combined >= best_score:
                best_score = combined
                save_checkpoint(args.ckpt_dir / "pro_stage2_best.pt", model, global_it, rec)

        report["history"].append(rec)
        print(
            f"[Stage2 {it:02d}/{args.iterations} | Iter {global_it}] games={total_games} len={rec['avg_game_len']} "
            f"p_loss={rec['policy_loss']:.3f} v_loss={rec['value_loss']:.3f} "
            f"({t_sp:.1f}s sp + {t_tr:.1f}s tr){eval_str}",
            flush=True,
        )

    best_ckpt_path = args.ckpt_dir / "pro_stage2_best.pt"
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        torch.save(best_ckpt, args.ckpt_dir / "pro.pt")

    # Re-benchmark Early / Mid / Pro consistently
    summary_tiers = {}
    for tier in ("early", "mid", "pro"):
        ckpt = torch.load(args.ckpt_dir / f"{tier}.pt", map_location="cpu", weights_only=False)
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

    report["elapsed_seconds"] = round(float(report["elapsed_seconds"]) + (time.perf_counter() - t_start), 1)
    report["total_selfplay_games"] = total_games
    report["total_augmented_samples"] = total_samples
    report["tiers"] = summary_tiers
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== Post-Stage2 Checkpoint Summary ===", flush=True)
    print(json.dumps(summary_tiers, indent=2), flush=True)


if __name__ == "__main__":
    main()
