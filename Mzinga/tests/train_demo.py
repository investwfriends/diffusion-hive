"""REINFORCE training demo for HiveEnv (100 episodes, scaled-down model).

~3 min. Observation compressed via fixed random projection (88 -> 16).
Intermediate rewards (surround, step bonus) anneal linearly from 1.0 -> 0.1
over training so the policy shifts from positional guidance to terminal outcomes.
"""

import time

import numpy as np

from mzinga.core.enums import Direction, GameType, PieceName, PlayerColor
from mzinga.gym.hive_env import HiveEnv, OBS_DIM

POLICY_DIM = 16

_rng = np.random.default_rng(seed=42)
_PROJ = _rng.standard_normal((POLICY_DIM, OBS_DIM)).astype(np.float32) * 0.1


class SparseSoftmaxPolicy:
    def __init__(self, lr: float = 0.02):
        self.W: dict[int, np.ndarray] = {}
        self.lr = lr
        self._rng = np.random.default_rng()

    def _weights(self, action: int) -> np.ndarray:
        if action not in self.W:
            self.W[action] = np.random.randn(POLICY_DIM).astype(np.float32) * 0.01
        return self.W[action]

    def _project(self, obs: np.ndarray) -> np.ndarray:
        return _PROJ @ obs

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        h = self._project(obs)
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return -1
        scores = np.array([self._weights(a) @ h for a in valid], dtype=np.float64)
        scores -= scores.max()
        probs = np.exp(scores) / np.exp(scores).sum()
        return int(self._rng.choice(valid, p=probs))

    def update(self, obs_list, act_list, mask_list, rew_list, gamma=0.95):
        T = len(rew_list)
        if T == 0:
            return

        returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in reversed(range(T)):
            G = rew_list[t] + gamma * G
            returns[t] = G
        baseline = returns.mean()

        grads: dict[int, np.ndarray] = {}

        for obs, a, mask, Adv in zip(obs_list, act_list, mask_list, returns - baseline):
            h = self._project(obs)
            valid = np.where(mask)[0]
            scores = np.array([self._weights(ai) @ h for ai in valid], dtype=np.float64)
            scores -= scores.max()
            probs = np.exp(scores) / np.exp(scores).sum()
            grad = -np.outer(probs, h)
            taken_idx = int(np.where(valid == a)[0][0])
            grad[taken_idx] += h
            for vi, ai in enumerate(valid):
                g = grads.setdefault(ai, np.zeros(POLICY_DIM, dtype=np.float32))
                g += float(Adv) * grad[vi] / T

        for ai, g in grads.items():
            self.W[ai] += self.lr * g


def _count_surround(board, queen_piece):
    queen_pos = board.get_position(queen_piece)
    if queen_pos.stack < 0:
        return 0
    return sum(
        1 for d in range(6)
        if board.get_piece_on_top_at(queen_pos.get_neighbor_at(Direction(d))) != PieceName.INVALID
    )


def run_episode(env, policy, max_steps=1000, step_bonus_scale=0.005, surround_scale=0.02, anneal_factor=1.0):
    obs, info = env.reset()
    obss, acts, masks, rews = [], [], [], []
    total = 0.0

    step_bonus = step_bonus_scale * anneal_factor
    surround = surround_scale * anneal_factor

    for step in range(1, max_steps + 1):
        a = policy.act(obs, info["action_mask"])
        if a < 0:
            break
        obss.append(obs)
        acts.append(a)
        masks.append(info["action_mask"])
        try:
            obs, r, term, trunc, info = env.step(a)
        except Exception:
            r = -2.0
            term = True
        r += (step / max_steps) * step_bonus
        if not term:
            enemy_queen = PieceName.wQ if env.board.current_color == PlayerColor.White else PieceName.bQ
            r += _count_surround(env.board, enemy_queen) * surround
        rews.append(r)
        total += r
        if term or trunc:
            break

    final_gs = env.board.get_game_string()[:40] if hasattr(env, 'board') else "N/A"
    final_state = env.board.board_state.name if hasattr(env, 'board') else "N/A"

    if len(rews) > 0:
        policy.update(obss, acts, masks, rews, 0.95)

    return total, len(rews), final_state, final_gs


def main():
    env = HiveEnv(game_type=GameType.Base, mode="self_play",
                  reward_config={"queen_placed_reward": 0.1,
                                 "noisy_move_reward": 0.05})
    policy = SparseSoftmaxPolicy(lr=0.02)

    n_episodes = 100
    all_rew = []
    all_len = []
    t0 = time.time()

    print(f"{'Ep':>7}  {'Return':>7}  {'Avg10':>7}  {'Steps':>5}  {'State':>14}  {'Anneal':>6}  {'Time':>5}")
    print("-" * 69)

    for ep in range(1, n_episodes + 1):
        anneal_factor = 1.0 - 0.9 * (ep - 1) / (n_episodes - 1)
        total, steps, state, gs = run_episode(env, policy, anneal_factor=anneal_factor)
        all_rew.append(total)
        all_len.append(steps)
        avg10 = np.mean(all_rew[-10:])
        elapsed = time.time() - t0

        print(f"[{ep:>2}/{n_episodes}]  {total:>+7.2f}  {avg10:>+7.2f}  {steps:>5}  {state:>14}  {anneal_factor:>6.2f}  {elapsed:>5.0f}s")

    elapsed = time.time() - t0
    print("-" * 69)
    print(f"Final avg return (last 10): {np.mean(all_rew[-10:]):+.3f}")
    print(f"Avg episode steps (last 10): {np.mean(all_len[-10:]):.0f}")
    print(f"Avg steps overall: {np.mean(all_len):.0f}")
    state_counts = {}
    # We don't track per-episode final state in a list, but close enough
    print(f"Total time: {elapsed:.1f}s")
    print(f"Policy memory: {len(policy.W)}/{2048} action weights trained")
    env.close()


if __name__ == "__main__":
    main()
