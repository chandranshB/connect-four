"""Batched AlphaZero MCTS and parallel self-play game generator for Connect Four."""

from __future__ import annotations

import math

import numpy as np
import torch

from c4_env import (
    CENTER_ORDER,
    NUM_ACTIONS,
    TOP_MASK,
    encode_obs_batch,
    is_draw,
    last_mover_won,
    play_move,
)


def _softmax_masked(logits: np.ndarray, legal_cols: list[int]) -> np.ndarray:
    priors = np.zeros(NUM_ACTIONS, dtype=np.float32)
    if not legal_cols:
        return priors
    legal_logits = logits[legal_cols]
    m = float(np.max(legal_logits))
    exp_l = np.exp(legal_logits - m)
    s = float(np.sum(exp_l))
    probs = exp_l / s if s > 0 else np.full(len(legal_cols), 1.0 / len(legal_cols), dtype=np.float32)
    for idx, c in enumerate(legal_cols):
        priors[c] = float(probs[idx])
    return priors


class MCTSNode:
    __slots__ = (
        "pos",
        "mask",
        "ply",
        "is_terminal",
        "terminal_value",
        "expanded",
        "legal_cols",
        "raw_priors",
        "priors",
        "visits",
        "value_sum",
        "total_visits",
        "children",
    )

    def __init__(
        self, pos: int, mask: int, ply: int, is_terminal: bool = False, terminal_value: float = 0.0
    ) -> None:
        self.pos = pos
        self.mask = mask
        self.ply = ply
        self.is_terminal = is_terminal
        self.terminal_value = terminal_value
        self.expanded = False
        self.legal_cols = [] if is_terminal else [c for c in CENTER_ORDER if (mask & TOP_MASK[c]) == 0]
        self.raw_priors: np.ndarray | None = None
        self.priors: np.ndarray | None = None
        self.visits = [0, 0, 0, 0, 0, 0, 0]
        self.value_sum = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.total_visits = 0
        self.children: dict[int, MCTSNode] = {}


def _make_node(pos: int, mask: int, ply: int) -> MCTSNode:
    if last_mover_won(pos, mask):
        return MCTSNode(pos, mask, ply, is_terminal=True, terminal_value=-1.0)
    if is_draw(ply):
        return MCTSNode(pos, mask, ply, is_terminal=True, terminal_value=0.0)
    return MCTSNode(pos, mask, ply)


def _backup(path: list[tuple[MCTSNode, int]], leaf_value: float) -> None:
    """Propagate leaf value (from leaf player's perspective) back up the path."""
    v = -leaf_value
    for node, action in reversed(path):
        node.visits[action] += 1
        node.value_sum[action] += v
        node.total_visits += 1
        v = -v


def _select_puct_action(node: MCTSNode, c_puct: float) -> int:
    sqrt_total = math.sqrt(node.total_visits if node.total_visits > 0 else 1)
    priors = node.priors
    visits = node.visits
    value_sum = node.value_sum
    best_col = node.legal_cols[0]
    best_score = -1e18

    for c in node.legal_cols:
        n = visits[c]
        q = (value_sum[c] / n) if n > 0 else 0.0
        u = c_puct * float(priors[c]) * sqrt_total / (1.0 + n)
        score = q + u
        if score > best_score:
            best_score = score
            best_col = c
    return best_col


@torch.inference_mode()
def _eval_nodes(
    model: torch.nn.Module, device: torch.device, nodes: list[MCTSNode]
) -> tuple[np.ndarray, np.ndarray]:
    states = [(n.pos, n.mask, n.ply) for n in nodes]
    obs = encode_obs_batch(states)
    x = torch.from_numpy(obs).to(device=device, non_blocking=True)
    logits_t, values_t = model(x)
    return logits_t.detach().float().cpu().numpy(), values_t.detach().float().cpu().numpy()


def _apply_root_noise(
    node: MCTSNode, rng: np.random.Generator, dirichlet_alpha: float, noise_eps: float
) -> None:
    assert node.raw_priors is not None
    if noise_eps <= 0.0 or len(node.legal_cols) <= 1:
        node.priors = node.raw_priors
        return
    noise = rng.dirichlet([dirichlet_alpha] * len(node.legal_cols))
    priors = node.raw_priors.copy()
    for idx, c in enumerate(node.legal_cols):
        priors[c] = (1.0 - noise_eps) * float(node.raw_priors[c]) + noise_eps * float(noise[idx])
    node.priors = priors


def run_batched_mcts(
    model: torch.nn.Module,
    device: torch.device,
    roots: list[MCTSNode],
    num_sims: int,
    c_puct: float = 1.5,
    add_noise: bool = False,
    dirichlet_alpha: float = 0.7,
    noise_eps: float = 0.25,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Run `num_sims` batched MCTS simulations for a list of non-terminal root nodes."""
    if not roots:
        return []
    if rng is None:
        rng = np.random.default_rng()

    unexpanded_roots = [r for r in roots if not r.expanded]
    if unexpanded_roots:
        logits, _ = _eval_nodes(model, device, unexpanded_roots)
        for i, root in enumerate(unexpanded_roots):
            root.raw_priors = _softmax_masked(logits[i], root.legal_cols)
            root.priors = root.raw_priors
            root.expanded = True

    if add_noise:
        for root in roots:
            _apply_root_noise(root, rng, dirichlet_alpha, noise_eps)
    else:
        for root in roots:
            root.priors = root.raw_priors

    if num_sims > 0:
        for _ in range(num_sims):
            pending_leaves: list[tuple[MCTSNode, list[tuple[MCTSNode, int]]]] = []
            for root in roots:
                if len(root.legal_cols) <= 1:
                    continue
                cur = root
                path: list[tuple[MCTSNode, int]] = []
                while cur.expanded and not cur.is_terminal:
                    a = _select_puct_action(cur, c_puct)
                    path.append((cur, a))
                    child = cur.children.get(a)
                    if child is None:
                        np_pos, np_mask, np_ply = play_move(cur.pos, cur.mask, cur.ply, a)
                        child = _make_node(np_pos, np_mask, np_ply)
                        cur.children[a] = child
                    cur = child

                if cur.is_terminal:
                    _backup(path, cur.terminal_value)
                else:
                    pending_leaves.append((cur, path))

            if pending_leaves:
                leaf_nodes = [item[0] for item in pending_leaves]
                logits, values = _eval_nodes(model, device, leaf_nodes)
                for i, (leaf, path) in enumerate(pending_leaves):
                    leaf.raw_priors = _softmax_masked(logits[i], leaf.legal_cols)
                    leaf.priors = leaf.raw_priors
                    leaf.expanded = True
                    _backup(path, float(values[i]))

    policies: list[np.ndarray] = []
    for root in roots:
        pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if len(root.legal_cols) == 1:
            pi[root.legal_cols[0]] = 1.0
        elif root.total_visits > 0:
            inv = 1.0 / root.total_visits
            for c in root.legal_cols:
                pi[c] = root.visits[c] * inv
        else:
            assert root.raw_priors is not None
            pi = root.raw_priors.copy()
        policies.append(pi)
    return policies


def generate_selfplay_batch(
    model: torch.nn.Module,
    device: torch.device,
    num_games: int = 128,
    num_sims: int = 64,
    temp_threshold_ply: int = 10,
    c_puct: float = 1.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Play `num_games` parallel self-play games and return 2x mirror-augmented training arrays."""
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
        policies = run_batched_mcts(
            model=model,
            device=device,
            roots=active_roots,
            num_sims=num_sims,
            c_puct=c_puct,
            add_noise=True,
            dirichlet_alpha=0.7,
            noise_eps=0.25,
            rng=rng,
        )

        next_active: list[int] = []
        for idx, pi in zip(active_indices, policies):
            node = roots[idx]
            player_sign = 1 if (node.ply % 2 == 0) else -1
            trajectories[idx].append(((node.pos, node.mask, node.ply), pi.copy(), player_sign))

            if node.ply < temp_threshold_ply:
                probs = pi.astype(np.float64)
                probs /= probs.sum()
                action = int(rng.choice(NUM_ACTIONS, p=probs))
            else:
                max_v = float(np.max(pi))
                best = [c for c in node.legal_cols if abs(float(pi[c]) - max_v) <= 1e-8]
                action = int(best[int(rng.integers(0, len(best)))])

            child = node.children.get(action)
            if child is None:
                np_pos, np_mask, np_ply = play_move(node.pos, node.mask, node.ply, action)
                child = _make_node(np_pos, np_mask, np_ply)

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

    stats = {
        "p1_wins": float(p1_wins),
        "p2_wins": float(p2_wins),
        "draws": float(draws),
        "avg_len": float(total_plies) / max(1, num_games),
    }
    return obs_aug, pis_aug, zs_aug, stats


class PolicyPlayer:
    """Fast agent playing either raw policy (0 sims) or MCTS-guided policy."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_sims: int = 64,
        c_puct: float = 1.5,
    ) -> None:
        self.model = model
        self.device = device
        self.num_sims = num_sims
        self.c_puct = c_puct

    @torch.inference_mode()
    def select_action(self, pos: int, mask: int, ply: int) -> int:
        self.model.eval()
        legal = [c for c in CENTER_ORDER if (mask & TOP_MASK[c]) == 0]
        if len(legal) == 1:
            return legal[0]
        if self.num_sims <= 0:
            obs = encode_obs_batch([(pos, mask, ply)])
            x = torch.from_numpy(obs).to(self.device)
            logits, _ = self.model(x)
            probs = _softmax_masked(logits[0].float().cpu().numpy(), legal)
            return max(legal, key=lambda c: float(probs[c]))

        root = MCTSNode(pos, mask, ply)
        policies = run_batched_mcts(
            model=self.model,
            device=self.device,
            roots=[root],
            num_sims=self.num_sims,
            c_puct=self.c_puct,
            add_noise=False,
        )
        pi = policies[0]
        return max(legal, key=lambda c: float(pi[c]))
