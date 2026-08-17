import asyncio
import pytest

from mzinga.core.board import Board
from mzinga.core.enums import BoardState, PieceName, GameType
from mzinga.core.move import Move
from mzinga.core.position import Position, ORIGIN_POSITION, NULL_POSITION

# --- Base Game Perft ---
PERFT_BASE = {
    0: 1,
    1: 4,
    2: 96,
    3: 1440,
    4: 21600,
}

@pytest.mark.parametrize("depth, expected", list(PERFT_BASE.items()))
def test_perft_base(depth, expected):
    b = Board(GameType.Base)
    result = asyncio.run(b.calculate_perft_async(depth))
    assert result == expected

# --- Base+M Game Perft ---
PERFT_BASE_M = {
    0: 1,
    1: 5,
    2: 150,
    3: 2610,
}

@pytest.mark.parametrize("depth, expected", list(PERFT_BASE_M.items()))
def test_perft_base_m(depth, expected):
    b = Board(GameType.BaseM)
    result = asyncio.run(b.calculate_perft_async(depth))
    assert result == expected

# --- Base+L Game Perft ---
PERFT_BASE_L = {
    0: 1,
    1: 5,
    2: 150,
    3: 2610,
}

@pytest.mark.parametrize("depth, expected", list(PERFT_BASE_L.items()))
def test_perft_base_l(depth, expected):
    b = Board(GameType.BaseL)
    result = asyncio.run(b.calculate_perft_async(depth))
    assert result == expected

# --- Base+P Game Perft ---
PERFT_BASE_P = {
    0: 1,
    1: 5,
    2: 150,
    3: 2610,
}

@pytest.mark.parametrize("depth, expected", list(PERFT_BASE_P.items()))
def test_perft_base_p(depth, expected):
    b = Board(GameType.BaseP)
    result = asyncio.run(b.calculate_perft_async(depth))
    assert result == expected

# --- Base+ML Game Perft ---
PERFT_BASE_ML = {
    0: 1,
    1: 6,
    2: 216,
    3: 4320,
}

@pytest.mark.parametrize("depth, expected", list(PERFT_BASE_ML.items()))
def test_perft_base_ml(depth, expected):
    b = Board(GameType.BaseML)
    result = asyncio.run(b.calculate_perft_async(depth))
    assert result == expected

# --- Base+MP Game Perft ---
PERFT_BASE_MP = {
    0: 1,
    1: 6,
    2: 216,
    3: 4320,
}

@pytest.mark.parametrize("depth, expected", list(PERFT_BASE_MP.items()))
def test_perft_base_mp(depth, expected):
    b = Board(GameType.BaseMP)
    result = asyncio.run(b.calculate_perft_async(depth))
    assert result == expected

# --- Base+LP Game Perft ---
PERFT_BASE_LP = {
    0: 1,
    1: 6,
    2: 216,
    3: 4320,
}

@pytest.mark.parametrize("depth, expected", list(PERFT_BASE_LP.items()))
def test_perft_base_lp(depth, expected):
    b = Board(GameType.BaseLP)
    result = asyncio.run(b.calculate_perft_async(depth))
    assert result == expected

# --- Base+MLP Game Perft ---
PERFT_BASE_MLP = {
    0: 1,
    1: 7,
    2: 294,
    3: 6678,
}

@pytest.mark.parametrize("depth, expected", list(PERFT_BASE_MLP.items()))
def test_perft_base_mlp(depth, expected):
    b = Board(GameType.BaseMLP)
    result = asyncio.run(b.calculate_perft_async(depth))
    assert result == expected
