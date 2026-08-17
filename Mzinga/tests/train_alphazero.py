"""AlphaZero-style training for Hive — MCTS-guided self-play + policy-value network.

~2h on CPU. Residual MLP (256 hidden, 4 blocks), MCTS with 50 simulations
per move, supervised learning from MCTS visit counts + game outcome regression.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from mzinga.core.board import Board
from mzinga.core.enums import BoardState, GameType, PlayerColor
from mzinga.gym.hive_env import MAX_MOVES, OBS_DIM
from mzinga.rl.mcts import MCTS, board_to_obs, game_outcome
from mzinga.rl.model import HivePolicyValue
from mzinga.rl.dashboard import TerminalDashboard

DEVICE = torch.device("cpu")


class ReplayBuffer:
    def __init__(self, max_size: int = 200_000):
        self.obs_list: list[np.ndarray] = []
        self.mask_list: list[np.ndarray] = []
        self.pi_list: list[np.ndarray] = []
        self.outcome_list: list[float] = []
        self.max_size = max_size

    def add_game(self, examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]):
        for obs, mask, pi, outcome in examples:
            if len(self.obs_list) >= self.max_size:
                idx = np.random.default_rng().integers(len(self.obs_list))
                self.obs_list[idx] = obs
                self.mask_list[idx] = mask
                self.pi_list[idx] = pi
                self.outcome_list[idx] = outcome
            else:
                self.obs_list.append(obs)
                self.mask_list.append(mask)
                self.pi_list.append(pi)
                self.outcome_list.append(outcome)

    def sample_batch(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        rng = np.random.default_rng()
        indices = rng.integers(len(self.obs_list), size=batch_size)
        obs_b = np.stack([self.obs_list[i] for i in indices])
        mask_b = np.stack([self.mask_list[i] for i in indices])
        pi_b = np.stack([self.pi_list[i] for i in indices])
        out_b = np.array([self.outcome_list[i] for i in indices], dtype=np.float32)
        return (
            torch.as_tensor(obs_b, device=device),
            torch.as_tensor(mask_b, dtype=torch.bool, device=device),
            torch.as_tensor(pi_b, device=device),
            torch.as_tensor(out_b, device=device),
        )

    def __len__(self):
        return len(self.obs_list)


def self_play_game(model, mcts: MCTS, max_steps: int = 300):
    board = Board(GameType.Base)
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    game_length = 0
    mcts_entropy_sum = 0.0
    mcts_entropy_n = 0

    for game_length in range(1, max_steps + 1):
        moves = list(board.get_valid_moves())
        if len(moves) == 0 or board.game_is_over:
            break

        pi, pi_probs, best_a = mcts.search(board)
        mcts_entropy_sum += mcts.root_visit_entropy()
        mcts_entropy_n += 1
        move = moves[best_a]
        move_str = board.try_get_move_string(move) or ""

        mask = np.zeros(MAX_MOVES, dtype=bool)
        mask[: len(moves)] = True

        obs = board_to_obs(board)
        examples.append((obs, mask, pi_probs, 0.0))

        board.trusted_play(move, move_str)

    terminated = board.game_is_over
    if terminated:
        outcome = game_outcome(board)
    else:
        outcome = 0.0

    for i in range(len(examples)):
        obs, mask, pi, _ = examples[i]
        is_black_turn = obs[84] > 0.0
        value = -outcome if is_black_turn else outcome
        examples[i] = (obs, mask, pi, value)

    mcts_entropy_avg = mcts_entropy_sum / max(1, mcts_entropy_n)
    return examples, board.board_state, game_length, terminated, mcts_entropy_avg


def train_step(
    model, optimizer, buffer: ReplayBuffer, batch_size: int, device: torch.device
) -> dict[str, float]:
    if len(buffer) < batch_size:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "grad_norm": 0.0,
            "policy_entropy": 0.0,
        }

    model.train()
    obs_b, mask_b, pi_b, out_b = buffer.sample_batch(batch_size, device)

    logits, values = model(obs_b)
    logits_m = logits.masked_fill(~mask_b, float("-inf"))
    log_probs = F.log_softmax(logits_m, dim=-1)

    policy_loss = -(pi_b * log_probs).masked_fill(~mask_b, 0.0).sum(-1).mean()
    value_loss = F.mse_loss(values, out_b)

    probs = F.softmax(logits_m, dim=-1)
    safe_log_p = log_probs.masked_fill(~mask_b, 0.0)
    _entropy = -(probs * safe_log_p).sum(-1).mean().item()

    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    grad_norm = gn.item() if hasattr(gn, "item") else float(gn)
    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "grad_norm": grad_norm,
        "policy_entropy": float(_entropy),
    }


def evaluate(model, n_games: int = 10, max_steps: int = 300):
    wins = 0
    losses = 0
    draws = 0

    for color in (PlayerColor.White, PlayerColor.Black):
        for _ in range(n_games // 2):
            board = Board(GameType.Base)
            for _ in range(max_steps):
                if board.game_is_over:
                    break
                moves = list(board.get_valid_moves())
                if len(moves) == 0:
                    break

                if board.current_color == color:
                    obs = board_to_obs(board)
                    mask = np.zeros(MAX_MOVES, dtype=bool)
                    mask[: len(moves)] = True
                    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=model.device)
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device)

                    with torch.no_grad():
                        logits, _ = model(obs_t.unsqueeze(0))
                        logits_m = logits.squeeze(0).masked_fill(~mask_t, float("-inf"))
                        action_idx = int(torch.argmax(logits_m))

                    move = moves[action_idx]
                    move_str = board.try_get_move_string(move) or ""
                    board.trusted_play(move, move_str)
                else:
                    opp_move = moves[np.random.default_rng().integers(len(moves))]
                    opp_str = board.try_get_move_string(opp_move) or ""
                    board.trusted_play(opp_move, opp_str)

            state = board.board_state
            if state == BoardState.Draw:
                draws += 1
            elif (
                (state == BoardState.WhiteWins and color == PlayerColor.White)
                or (state == BoardState.BlackWins and color == PlayerColor.Black)
            ):
                wins += 1
            else:
                losses += 1

    return wins / n_games, losses / n_games, draws / n_games


def main():
    print(f"Device: {DEVICE}")
    model = HivePolicyValue(obs_dim=OBS_DIM, hidden_dim=256, num_blocks=4)
    model.to(DEVICE)
    print(f"Parameters: {model.param_count():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-5)

    mcts = MCTS(
        model=model,
        num_simulations=30,
        c_puct=1.4,
        temperature=1.0,
        temperature_threshold=15,
    )

    buffer = ReplayBuffer(max_size=200_000)

    n_iterations = 250
    games_per_iter = 2
    batch_size = 256
    train_epochs_per_iter = 3

    dashboard = TerminalDashboard(
        n_iterations=n_iterations,
        title="AlphaZero — 847K params, 30 sims, cpu",
        log_path="training_metrics.jsonl",
    )

    total_games = 0
    total_moves = 0
    total_terminated = 0
    win_rate = 0.0
    eval_wins = eval_losses = eval_draws = None

    for it in range(1, n_iterations + 1):
        it_mcts_ent = 0.0
        mcts_ent_cnt = 0

        for _ in range(games_per_iter):
            examples, final_state, game_len, term, mcts_ent = self_play_game(model, mcts)
            buffer.add_game(examples)
            total_games += 1
            total_moves += game_len
            total_terminated += int(term)
            if mcts_ent > 0:
                it_mcts_ent += mcts_ent
                mcts_ent_cnt += 1

        mcts_ent_avg = it_mcts_ent / max(1, mcts_ent_cnt)

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "grad_norm": 0.0,
            "policy_entropy": 0.0,
        }
        for _ in range(train_epochs_per_iter):
            metrics = train_step(model, optimizer, buffer, batch_size, DEVICE)
        scheduler.step()

        if it % 30 == 0 or it == 1:
            w, l, d = evaluate(model, n_games=10)
            win_rate = w
            eval_wins = int(w * 10)
            eval_losses = int(l * 10)
            eval_draws = int(d * 10)

        avg_moves = total_moves / max(1, total_games)
        term_rate = total_terminated / max(1, total_games)

        dashboard.update(
            it,
            win_rate=win_rate,
            policy_loss=metrics["policy_loss"],
            value_loss=metrics["value_loss"],
            grad_norm=metrics["grad_norm"],
            policy_entropy=metrics["policy_entropy"],
            mcts_entropy=mcts_ent_avg,
            buf_size=len(buffer),
            lr=scheduler.get_last_lr()[0],
            avg_moves=avg_moves,
            terminal_rate=term_rate,
            total_games=total_games,
            eval_wins=eval_wins,
            eval_losses=eval_losses,
            eval_draws=eval_draws,
        )

    dashboard.close()

    elapsed = time.time() - dashboard._start_time
    final_win, final_loss, final_draw = evaluate(model, n_games=50)
    print(f"\nFinal evaluation (50 games vs random):")
    print(f"  Wins: {final_win:.1%}  Losses: {final_loss:.1%}  Draws: {final_draw:.1%}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print(f"  Parameters: {model.param_count():,}")
    print(f"  Buffer size: {len(buffer):,}")
    print(f"  Total games: {total_games}")

    torch.save(model.state_dict(), "mzinga_alphazero.pt")
    print("  Model saved: mzinga_alphazero.pt")


if __name__ == "__main__":
    main()
