"""PPO self-play training for Hive — scaled-up policy-value network.

~2h on CPU. Residual MLP (256 hidden, 4 blocks), PPO-CLIP with GAE,
±10 terminal rewards, surround-progress bonus annealed over training.
Self-play with HiveEnv (Base), evaluated vs random opponent.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from mzinga.core.enums import BoardState, Direction, GameType, PieceName, PlayerColor
from mzinga.gym.hive_env import HiveEnv, OBS_DIM
from mzinga.rl.model import HivePolicyValue
from mzinga.rl.ppo import RolloutBuffer, ppo_update

DEVICE = torch.device("cpu")  # use "mps" for Apple Silicon if stable


def _count_surround(board, queen_piece):
    queen_pos = board.get_position(queen_piece)
    if queen_pos.stack < 0:
        return 0
    return sum(
        1
        for d in range(6)
        if board.get_piece_on_top_at(queen_pos.get_neighbor_at(Direction(d)))
        != PieceName.INVALID
    )


def _make_env():
    return HiveEnv(
        game_type=GameType.Base,
        mode="self_play",
        reward_config={
            "win_reward": 10.0,
            "loss_reward": -10.0,
        },
    )


def _collect_rollout(model, buffer, steps_per_iter, surround_scale=0.01):
    env = _make_env()
    obs, info = env.reset()
    steps_collected = 0

    while steps_collected < steps_per_iter:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
        mask_t = torch.as_tensor(info["action_mask"], dtype=torch.bool, device=model.device)
        action, log_prob, value = model.act(obs_t, mask_t)

        try:
            next_obs, reward, terminated, truncated, info = env.step(action)
        except Exception:
            reward = -2.0
            truncated = True
            next_obs, _, terminated, _, info = env.reset()

        if not terminated:
            enemy_queen = (
                PieceName.wQ
                if env.board.current_color == PlayerColor.White
                else PieceName.bQ
            )
            reward += _count_surround(env.board, enemy_queen) * surround_scale

        done = terminated or truncated

        buffer.add(
            obs,
            action,
            log_prob.cpu().item(),
            value.cpu().item(),
            reward,
            mask_t.cpu().numpy(),
            done,
        )
        steps_collected += 1

        if done:
            obs, info = env.reset()
        else:
            obs = next_obs

    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
    _, last_value = model(obs_t.unsqueeze(0))
    return last_value.item()


def _evaluate(model, n_games=10, max_steps=300):
    wins = 0
    losses = 0
    draws = 0

    for color in (PlayerColor.White, PlayerColor.Black):
        for _ in range(n_games // 2):
            env = HiveEnv(
                game_type=GameType.Base,
                mode="vs_opponent",
                agent_color=color,
                opponent="random",
            )
            obs, info = env.reset()
            for _ in range(max_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
                mask_t = torch.as_tensor(info["action_mask"], dtype=torch.bool, device=model.device)
                action, _, _ = model.act(obs_t, mask_t, deterministic=True)

                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                except Exception:
                    losses += 1
                    break

                if terminated or truncated:
                    state = env.board.board_state
                    if state == BoardState.Draw:
                        draws += 1
                    elif (
                        (state == BoardState.WhiteWins and color == PlayerColor.White)
                        or (state == BoardState.BlackWins and color == PlayerColor.Black)
                    ):
                        wins += 1
                    else:
                        losses += 1
                    break
            else:
                draws += 1

    return wins / n_games, losses / n_games, draws / n_games


def main():
    print(f"Device: {DEVICE}")
    model = HivePolicyValue(obs_dim=OBS_DIM, hidden_dim=256, num_blocks=4)
    model.to(DEVICE)
    print(f"Parameters: {model.param_count():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.02, total_iters=2000)

    n_iterations = 2000
    steps_per_iter = 1024
    n_epochs = 6
    batch_size = 128
    clip_epsilon = 0.15
    ent_coef = 0.005
    vf_coef = 1.0
    gamma = 0.998
    gae_lambda = 0.95

    buffer = RolloutBuffer(OBS_DIM, steps_per_iter)

    t0 = time.time()
    print(
        f"{'Iter':>5}  {'PolicyLoss':>10}  {'ValueLoss':>10}  {'Entropy':>8}  "
        f"{'WinRate':>7}  {'StepsAvg':>8}  {'LR':>8}  {'Time':>6}"
    )
    print("-" * 80)

    step_counts = []

    for it in range(1, n_iterations + 1):
        surround_scale = 0.005 * max(0.1, 1.0 - it / n_iterations)

        last_val = _collect_rollout(model, buffer, steps_per_iter, surround_scale)
        step_counts.append(len(buffer))
        buffer.compute_gae(last_val, gamma, gae_lambda)

        metrics = ppo_update(
            model, optimizer, buffer, batch_size, n_epochs,
            clip_epsilon, ent_coef, vf_coef, DEVICE,
        )
        buffer.clear()
        scheduler.step()

        if it % 50 == 0 or it == 1:
            win_rate, loss_rate, draw_rate = _evaluate(model, n_games=10)
            avg_steps = np.mean(step_counts[-25:]) if step_counts else 0
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(
                f"{it:>5}  {metrics['policy_loss']:>10.4f}  "
                f"{metrics['value_loss']:>10.4f}  {metrics['entropy']:>8.4f}  "
                f"{win_rate:>6.2f}  {avg_steps:>8.0f}  {lr:>8.2e}  {elapsed:>5.0f}s"
            )

    elapsed = time.time() - t0
    print("-" * 80)
    final_win, final_loss, final_draw = _evaluate(model, n_games=50)
    print(f"\nFinal evaluation (50 games):")
    print(f"  Wins: {final_win:.1%}  Losses: {final_loss:.1%}  Draws: {final_draw:.1%}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print(f"  Parameters: {model.param_count():,}")

    torch.save(model.state_dict(), "mzinga_ppo.pt")
    print("  Model saved: mzinga_ppo.pt")


if __name__ == "__main__":
    main()
