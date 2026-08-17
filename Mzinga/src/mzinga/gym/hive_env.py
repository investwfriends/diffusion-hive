from __future__ import annotations

from typing import Any, Callable, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mzinga.core.board import Board, InvalidMoveError
from mzinga.core.enums import (
    BoardState,
    BugType,
    GameType,
    PieceName,
    PlayerColor,
    game_is_over,
    game_in_progress,
    get_bug_type,
)
from mzinga.core.move import Move, PASS_MOVE
from mzinga.core.position import NULL_POSITION

MAX_MOVES = 2048
OBS_DIM = 88
BOARD_NORM = 16.0

_RewardConfig = dict[str, float]
_OpponentFn = Callable[[Board, list[Move]], Move]


class HiveEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        game_type: GameType = GameType.Base,
        mode: str = "self_play",
        agent_color: PlayerColor = PlayerColor.White,
        opponent: Union[str, _OpponentFn] = "random",
        reward_config: Optional[_RewardConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self._game_type = game_type
        self._mode = mode
        self._agent_color = agent_color
        self._opponent = opponent
        self._reward_config: _RewardConfig = reward_config or {}
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(MAX_MOVES)

        self._board: Optional[Board] = None

    @property
    def board(self) -> Board:
        if self._board is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        return self._board

    @property
    def game_type(self) -> GameType:
        return self._game_type

    @property
    def _opponent_color(self) -> PlayerColor:
        if self._agent_color == PlayerColor.White:
            return PlayerColor.Black
        return PlayerColor.White

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._board = Board(self._game_type)

        if self._mode == "vs_opponent" and self._agent_color == PlayerColor.Black:
            self._play_opponent_move()

        obs = self._build_observation()
        info = self._build_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        valid_moves_list = list(self.board.get_valid_moves())

        if action < 0 or action >= len(valid_moves_list):
            obs = self._build_observation()
            reward = self._reward_config.get("invalid_action_penalty", -0.01)
            info = self._build_info()
            return obs, reward, False, False, info

        move = valid_moves_list[action]
        move_str = self.board.try_get_move_string(move) or ""
        prev_color = self.board.current_color

        try:
            self.board.play(move, move_str)
        except InvalidMoveError:
            obs = self._build_observation()
            reward = self._reward_config.get("invalid_action_penalty", -0.01)
            info = self._build_info()
            return obs, reward, False, False, info

        intermediate_reward = self._compute_intermediate_reward(move)

        if self.board.game_is_over:
            terminal_reward = self._compute_terminal_reward(prev_color)
            obs = self._build_observation()
            info = self._build_info()
            return obs, terminal_reward + intermediate_reward, True, False, info

        if self._mode == "vs_opponent":
            self._play_opponent_move()
            if self.board.game_is_over:
                terminal_reward = self._compute_terminal_reward(self._opponent_color)
                obs = self._build_observation()
                info = self._build_info()
                return obs, terminal_reward + intermediate_reward, True, False, info

        obs = self._build_observation()
        info = self._build_info()
        return obs, intermediate_reward, False, False, info

    def action_masks(self) -> np.ndarray:
        valid_moves = self.board.get_valid_moves()
        mask = np.zeros(MAX_MOVES, dtype=bool)
        mask[: len(valid_moves)] = True
        return mask

    def render(self) -> Optional[str]:
        if self.render_mode == "ansi":
            board = self.board
            lines = [
                f"Turn: {board.current_turn} ({board.current_color.name})",
                f"State: {board.board_state.name}",
                f"Game type: {board.game_type.name}",
                f"Game string: {board.get_game_string()}",
                "",
                "Pieces in play:",
            ]
            for pn in PieceName:
                if pn in (PieceName.INVALID, PieceName.NumPieceNames):
                    continue
                pos = board.get_position(pn)
                if pos.stack >= 0:
                    lines.append(f"  {pn.name}: ({pos.q}, {pos.r}, stack={pos.stack})")
            return "\n".join(lines)
        return None

    def _build_observation(self) -> np.ndarray:
        board = self.board
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        for pn in PieceName:
            if pn in (PieceName.INVALID, PieceName.NumPieceNames):
                continue
            idx = pn.value * 3
            pos = board.get_position(pn)
            if pos.stack < 0:
                obs[idx] = 0.0
                obs[idx + 1] = 0.0
                obs[idx + 2] = -1.0
            else:
                obs[idx] = float(np.clip(pos.q / BOARD_NORM, -1.0, 1.0))
                obs[idx + 1] = float(np.clip(pos.r / BOARD_NORM, -1.0, 1.0))
                obs[idx + 2] = pos.stack / 8.0

        obs[84] = 1.0 if board.current_color == PlayerColor.Black else -1.0
        obs[85] = board.current_turn / 100.0
        obs[86] = 1.0 if board.current_turn_queen_in_play else 0.0
        obs[87] = 1.0 if board.game_is_over else 0.0

        return obs

    def _build_info(self) -> dict:
        mask = self.action_masks()
        return {"action_mask": mask, "moves_count": int(mask.sum())}

    def _compute_terminal_reward(self, player_color: PlayerColor) -> float:
        board = self.board
        if board.board_state == BoardState.Draw:
            return self._reward_config.get("draw_reward", 0.0)

        if self._mode == "self_play":
            player_won = (
                player_color == PlayerColor.White
                and board.board_state == BoardState.WhiteWins
            ) or (
                player_color == PlayerColor.Black
                and board.board_state == BoardState.BlackWins
            )
            if player_won:
                return self._reward_config.get("win_reward", 1.0)
            return self._reward_config.get("loss_reward", -1.0)

        agent_won = (
            self._agent_color == PlayerColor.White
            and board.board_state == BoardState.WhiteWins
        ) or (
            self._agent_color == PlayerColor.Black
            and board.board_state == BoardState.BlackWins
        )
        if agent_won:
            return self._reward_config.get("win_reward", 1.0)
        return self._reward_config.get("loss_reward", -1.0)

    def _compute_intermediate_reward(self, move: Move) -> float:
        if move == PASS_MOVE:
            return self._reward_config.get("pass_move_reward", 0.0)

        total = 0.0
        bug_type = get_bug_type(move.piece_name)
        if bug_type == BugType.QueenBee:
            total += self._reward_config.get("queen_placed_reward", 0.0)

        if self.board.is_noisy_move(move):
            total += self._reward_config.get("noisy_move_reward", 0.0)

        return total

    def _play_opponent_move(self) -> None:
        if self.board.game_is_over:
            return
        valid_moves = self.board.get_valid_moves()
        valid_moves_list = list(valid_moves)

        if callable(self._opponent):
            move = self._opponent(self.board, valid_moves)
        elif self._opponent == "random":
            if len(valid_moves_list) == 0:
                return
            idx = self.np_random.integers(len(valid_moves_list))
            move = valid_moves_list[idx]
        elif self._opponent == "pass":
            if PASS_MOVE in valid_moves:
                move = PASS_MOVE
            else:
                move = valid_moves_list[0]
        else:
            raise ValueError(f"Unknown opponent type: {self._opponent}")

        move_str = self.board.try_get_move_string(move) or ""
        self.board.trusted_play(move, move_str)
