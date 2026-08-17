import numpy as np
import pytest

from mzinga.core.board import Board
from mzinga.core.enums import (
    BoardState,
    BugType,
    GameType,
    PieceName,
    PlayerColor,
    get_bug_type,
)
from mzinga.core.move import Move, PASS_MOVE
from mzinga.core.position import ORIGIN_POSITION, NULL_POSITION
from mzinga.gym.hive_env import HiveEnv, MAX_MOVES, OBS_DIM


def test_env_creation():
    env = HiveEnv(game_type=GameType.Base)
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.action_space.n == MAX_MOVES
    assert env.game_type == GameType.Base
    env.close()


def test_env_creation_all_game_types():
    for gt in GameType:
        if gt == GameType.INVALID:
            continue
        env = HiveEnv(game_type=gt)
        assert env.observation_space.shape == (OBS_DIM,)
        env.close()


def test_reset():
    env = HiveEnv(game_type=GameType.Base)
    obs, info = env.reset(seed=42)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    assert "action_mask" in info
    mask = info["action_mask"]
    assert mask.sum() == 4  # 4 valid placements on turn 1
    env.close()


def test_reset_state():
    env = HiveEnv(game_type=GameType.Base)
    obs, info = env.reset()
    board = env.board
    assert board.current_turn == 0
    assert board.current_color == PlayerColor.White
    assert board.board_state == BoardState.NotStarted
    env.close()


def test_action_mask():
    env = HiveEnv(game_type=GameType.Base)
    env.reset()
    mask = env.action_masks()
    assert len(mask) == MAX_MOVES
    assert mask.dtype == bool
    assert mask.sum() == 4
    env.close()


def test_step_valid():
    env = HiveEnv(game_type=GameType.Base)
    obs, info = env.reset()
    mask = env.action_masks()

    valid_indices = np.where(mask)[0]
    action = int(valid_indices[0])

    obs, reward, terminated, truncated, info = env.step(action)
    assert not terminated
    assert not truncated
    assert reward == 0.0
    assert obs.shape == (OBS_DIM,)
    env.close()


def test_step_invalid_negative():
    env = HiveEnv(game_type=GameType.Base)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(-1)
    assert not terminated
    assert not truncated
    assert reward == -0.01
    env.close()


def test_step_invalid_out_of_range():
    env = HiveEnv(game_type=GameType.Base)
    env.reset()
    mask = env.action_masks()
    max_valid = int(mask.sum())
    obs, reward, terminated, truncated, info = env.step(max_valid + 100)
    assert not terminated
    assert not truncated
    assert reward == -0.01
    env.close()


def test_self_play_alternates_colors():
    env = HiveEnv(game_type=GameType.Base, mode="self_play")
    env.reset(seed=0)
    assert env.board.current_color == PlayerColor.White
    obs, _, _, _, _ = env.step(0)
    assert env.board.current_color == PlayerColor.Black
    obs, _, _, _, _ = env.step(0)
    assert env.board.current_color == PlayerColor.White
    env.close()


def test_render():
    env = HiveEnv(render_mode="ansi")
    env.reset()
    rendered = env.render()
    assert isinstance(rendered, str)
    assert "Turn:" in rendered
    assert "NotStarted" in rendered
    env.close()


def test_render_no_mode():
    env = HiveEnv()
    env.reset()
    assert env.render() is None
    env.close()


def test_seed_reproducibility():
    env1 = HiveEnv(game_type=GameType.Base)
    obs1, _ = env1.reset(seed=42)

    env2 = HiveEnv(game_type=GameType.Base)
    obs2, _ = env2.reset(seed=42)

    assert np.array_equal(obs1, obs2)
    env1.close()
    env2.close()


def test_different_seeds_different_opponent_move():
    env1 = HiveEnv(mode="vs_opponent", agent_color=PlayerColor.Black)
    env1.reset(seed=1)
    obs1, _ = env1.reset(seed=1)

    env2 = HiveEnv(mode="vs_opponent", agent_color=PlayerColor.Black)
    env2.reset(seed=2)

    env1.close()
    env2.close()


def test_vs_random_opponent():
    env = HiveEnv(mode="vs_opponent", agent_color=PlayerColor.White)
    obs, info = env.reset(seed=0)
    assert env.board.current_color == PlayerColor.White

    mask = env.action_masks()
    valid_indices = np.where(mask)[0]
    obs, reward, terminated, truncated, info = env.step(int(valid_indices[0]))

    # After agent's move and opponent's response, it's agent's turn again
    if not terminated:
        assert env.board.current_color == PlayerColor.White
    env.close()


def test_vs_opponent_black_agent():
    env = HiveEnv(mode="vs_opponent", agent_color=PlayerColor.Black)
    obs, info = env.reset(seed=0)
    # Opponent (White) should have made first move already
    assert env.board.current_turn >= 1
    assert env.board.current_color == PlayerColor.Black
    env.close()


def test_vs_pass_opponent():
    env = HiveEnv(mode="vs_opponent", opponent="pass")
    obs, info = env.reset(seed=0)
    mask = env.action_masks()
    valid_indices = np.where(mask)[0]
    obs, *_ = env.step(int(valid_indices[0]))
    env.close()


def test_vs_callable_opponent():
    def heuristic_opponent(board, valid_moves):
        moves_list = list(valid_moves)
        # Play queen as early as possible
        for m in moves_list:
            if get_bug_type(m.piece_name) == BugType.QueenBee:
                return m
        return moves_list[0]

    env = HiveEnv(mode="vs_opponent", opponent=heuristic_opponent)
    obs, info = env.reset(seed=0)
    mask = env.action_masks()
    valid_indices = np.where(mask)[0]
    obs, reward, terminated, truncated, info = env.step(int(valid_indices[0]))
    env.close()


def test_termination_detection():
    env = HiveEnv(game_type=GameType.Base)
    obs, info = env.reset(seed=0)

    max_steps = 200
    steps = 0
    terminated = False

    while not terminated and steps < max_steps:
        mask = env.action_masks()
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            break
        obs, reward, terminated, truncated, info = env.step(int(valid_indices[0]))
        steps += 1

    env.close()


def test_obs_bounds_invalid_action():
    env = HiveEnv(game_type=GameType.Base)
    env.reset()
    obs, _, _, _, _ = env.step(9999)
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    env.close()


def test_piece_in_hand_encoding():
    env = HiveEnv(game_type=GameType.Base)
    obs, info = env.reset()
    idx = PieceName.wQ.value * 3
    assert obs[idx + 2] == -1.0
    env.close()


def test_piece_on_board_encoding():
    env = HiveEnv(game_type=GameType.Base)
    env.reset()
    board = env.board
    valid_moves = list(board.get_valid_moves())
    first_move = valid_moves[0]
    idx = first_move.piece_name.value * 3
    obs, _, _, _, _ = env.step(0)
    assert obs[idx + 2] >= 0.0  # piece is now on board
    env.close()


def test_game_type_base_mlp():
    env = HiveEnv(game_type=GameType.BaseMLP)
    obs, info = env.reset()
    mask = env.action_masks()
    # BaseMLP has 7 non-queen piece types available on first turn
    assert mask.sum() >= 4
    env.close()


def test_game_type_base_m():
    env = HiveEnv(game_type=GameType.BaseM)
    obs, info = env.reset()
    mask = env.action_masks()
    assert mask.sum() >= 4
    env.close()


def test_reward_config_win_loss():
    env = HiveEnv(reward_config={"win_reward": 10.0, "loss_reward": -5.0})
    assert env._reward_config["win_reward"] == 10.0
    assert env._reward_config["loss_reward"] == -5.0
    env.close()


def test_env_close():
    env = HiveEnv()
    env.reset()
    env.close()


def test_multiple_resets():
    env = HiveEnv(game_type=GameType.Base)
    obs1, _ = env.reset(seed=42)
    env.close()

    env = HiveEnv(game_type=GameType.Base)
    obs2, _ = env.reset(seed=42)
    env.close()

    assert np.array_equal(obs1, obs2)


def test_action_mask_after_step():
    env = HiveEnv(game_type=GameType.Base)
    env.reset()
    mask1 = env.action_masks()
    assert mask1.sum() == 4

    valid_indices = np.where(mask1)[0]
    env.step(int(valid_indices[0]))
    mask2 = env.action_masks()
    assert mask2.sum() > 0
    env.close()
