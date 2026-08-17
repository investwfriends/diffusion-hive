from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from mzinga.core.board import Board
from mzinga.core.enums import BoardState, PieceName, PlayerColor
from mzinga.core.move import PASS_MOVE
from mzinga.gym.hive_env import BOARD_NORM, MAX_MOVES, OBS_DIM

_C_TERMINAL = -1


def board_to_obs(board: Board) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    for pn in PieceName:
        if pn in (PieceName.INVALID, PieceName.NumPieceNames):
            continue
        idx = pn.value * 3
        pos = board.get_position(pn)
        if pos.stack < 0:
            obs[idx] = 0.0
            obs[idx + 1] = 0.0
            obs[idx + 2] = _C_TERMINAL
        else:
            obs[idx] = float(np.clip(pos.q / BOARD_NORM, -1.0, 1.0))
            obs[idx + 1] = float(np.clip(pos.r / BOARD_NORM, -1.0, 1.0))
            obs[idx + 2] = pos.stack / 8.0
    obs[84] = 1.0 if board.current_color == PlayerColor.Black.value else -1.0
    obs[85] = board.current_turn / 100.0
    obs[86] = 1.0 if board.current_turn_queen_in_play else 0.0
    obs[87] = 1.0 if board.game_is_over else 0.0
    return obs


def _board_terminal_value(board: Board) -> float:
    state = board.board_state
    if state == BoardState.Draw:
        return 0.0
    if state == BoardState.WhiteWins:
        return 1.0 if board.current_color == PlayerColor.White else -1.0
    if state == BoardState.BlackWins:
        return 1.0 if board.current_color == PlayerColor.Black else -1.0
    return 0.0


def game_outcome(board: Board) -> float:
    state = board.board_state
    if state == BoardState.Draw:
        return 0.0
    if state == BoardState.WhiteWins:
        return 1.0
    if state == BoardState.BlackWins:
        return -1.0
    return 0.0


class MCTS:
    def __init__(
        self,
        model: torch.nn.Module,
        num_simulations: int = 50,
        c_puct: float = 1.4,
        temperature: float = 1.0,
        temperature_threshold: int = 30,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.tree: dict[int, dict[int, tuple[float, float, float, float]]] = {}
        self._sim_path: list[tuple[int, int]] = []

    def search(
        self, board: Board
    ) -> tuple[np.ndarray, np.ndarray, float]:
        root_key = board.zobrist_key
        self._root_key = root_key
        self.tree = {}

        root_moves = board.get_valid_moves()
        n_actions = len(root_moves)

        if n_actions == 0:
            return np.zeros(MAX_MOVES, dtype=np.float32), np.zeros(MAX_MOVES, dtype=np.float32), 0.0

        model_device = next(self.model.parameters()).device
        root_obs_t = torch.as_tensor(
            board_to_obs(board), dtype=torch.float32, device=model_device
        )
        root_mask = self._build_mask(root_moves)

        with torch.no_grad():
            root_logits, _ = self.model(root_obs_t.unsqueeze(0))
            root_logits_m = root_logits.squeeze(0).masked_fill(~root_mask, float("-inf"))
            root_priors = F.softmax(root_logits_m, dim=-1).cpu().numpy()

        root_priors_noisy = self._add_dirichlet(root_priors, root_mask.cpu().numpy(), n_actions)

        self.tree[root_key] = {}
        for ai in range(MAX_MOVES):
            if root_mask[ai]:
                self.tree[root_key][ai] = (0.0, 0.0, float(root_priors_noisy[ai]), float(root_priors[ai]))

        for _ in range(self.num_simulations):
            self._sim_path.clear()
            sim_key = root_key

            while sim_key in self.tree and self.tree[sim_key] and not board.game_is_over:
                children = self.tree[sim_key]
                best_a = None
                best_ucb = -float("inf")
                total_N = sum(c[0] for c in children.values())

                for a, (N, W, P_current, P_root) in children.items():
                    Q = W / N if N > 0 else 0.0
                    ucb = Q + self.c_puct * P_current * math.sqrt(total_N + 1e-6) / (1.0 + N)
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_a = a

                if best_a is None:
                    break

                sim_moves = board.get_valid_moves()
                if best_a >= len(sim_moves):
                    break

                self._sim_path.append((sim_key, best_a))
                # Pass empty string to trusted_play: move_str is only used for
                # board_history and game-string serialization, which MCTS doesn't read.
                board.trusted_play(sim_moves[best_a], "")
                sim_key = board.zobrist_key

            if board.game_is_over:
                leaf_value = _board_terminal_value(board)
            else:
                sim_moves = board.get_valid_moves()
                if len(sim_moves) > 0:
                    sim_obs_t = torch.as_tensor(
                        board_to_obs(board), dtype=torch.float32, device=model_device
                    )
                    sim_mask = self._build_mask(sim_moves)
                    with torch.no_grad():
                        sim_logits, sim_value = self.model(sim_obs_t.unsqueeze(0))
                        sim_logits_m = sim_logits.squeeze(0).masked_fill(
                            ~sim_mask, float("-inf")
                        )
                        sim_priors = F.softmax(sim_logits_m, dim=-1).cpu().numpy()
                    if sim_key not in self.tree:
                        self.tree[sim_key] = {}
                    for ai in range(MAX_MOVES):
                        if sim_mask[ai] and ai not in self.tree[sim_key]:
                            self.tree[sim_key][ai] = (0.0, 0.0, float(sim_priors[ai]), float(sim_priors[ai]))
                    leaf_value = sim_value.item()
                else:
                    leaf_value = 0.0

            for prev_key, action in reversed(self._sim_path):
                N, W, P_current, P_root = self.tree[prev_key][action]
                N += 1.0
                W += leaf_value
                self.tree[prev_key][action] = (N, W, P_current, P_root)
                leaf_value = -leaf_value

            for _ in range(len(self._sim_path)):
                self._undo(board)

        children = self.tree[root_key]
        n_visits = {}
        for a, (N, W, P_current, P_root) in children.items():
            if N > 0:
                n_visits[a] = N

        if not n_visits:
            pi = np.zeros(MAX_MOVES, dtype=np.float32)
            pi_probs = root_priors_noisy
            best_a = int(np.argmax(root_priors))
        else:
            temp = self.temperature if board.current_turn < self.temperature_threshold else 1e-8
            indices = []
            visits_list = []
            for ai in sorted(n_visits.keys()):
                indices.append(ai)
                N = n_visits[ai]
                if temp > 1e-8:
                    visits_list.append(N ** (1.0 / temp))
                else:
                    visits_list.append(N)
            visits_arr = np.array(visits_list, dtype=np.float64)
            probs = visits_arr / visits_arr.sum()
            pi = np.zeros(MAX_MOVES, dtype=np.float32)
            pi_probs = np.zeros(MAX_MOVES, dtype=np.float32)
            for i, ai in enumerate(indices):
                pi[ai] = probs[i]
                pi_probs[ai] = N / sum(n_visits.values())
            best_a = max(n_visits, key=lambda k: n_visits[k])

        return pi, pi_probs, best_a

    @staticmethod
    def _undo(board: Board) -> None:
        item = board.board_history._items.pop()
        move = item.move
        if move != PASS_MOVE:
            board._set_position(move.piece_name, move.source, True)
        board._current_turn = board._current_turn - 1
        board._reset_caches()

    def _build_mask(self, moves):
        mask = np.zeros(MAX_MOVES, dtype=bool)
        mask[: len(moves)] = True
        return torch.as_tensor(mask, dtype=torch.bool, device=self.model.device)

    def root_visit_entropy(self) -> float:
        if not hasattr(self, "_root_key") or self._root_key not in self.tree:
            return 0.0
        children = self.tree[self._root_key]
        visits = [c[0] for c in children.values() if c[0] > 0]
        if not visits:
            return 0.0
        total = sum(visits)
        probs = [v / total for v in visits]
        return -sum(p * math.log(max(p, 1e-12)) for p in probs)

    def _add_dirichlet(self, priors, mask, n_valid):
        if n_valid <= 1:
            return priors
        noise = np.random.default_rng().dirichlet([self.dirichlet_alpha] * n_valid)
        result = priors.copy()
        j = 0
        for ai in range(MAX_MOVES):
            if mask[ai]:
                result[ai] = (1.0 - self.dirichlet_epsilon) * priors[ai] + self.dirichlet_epsilon * noise[j]
                j += 1
        return result
