from mzinga.core.position import Position, ORIGIN_POSITION, NULL_POSITION, ORIGIN_NEIGHBORS, NEIGHBOR_DELTAS
from mzinga.core.enums import Direction


def test_origin_position():
    assert ORIGIN_POSITION.q == 0
    assert ORIGIN_POSITION.r == 0
    assert ORIGIN_POSITION.stack == 0


def test_null_position():
    assert NULL_POSITION.q == 0
    assert NULL_POSITION.r == 0
    assert NULL_POSITION.stack == -1


def test_equality():
    assert Position(1, 2, 3) == Position(1, 2, 3)


def test_inequality():
    assert Position(1, 2, 3) != Position(1, 2, 4)


def test_origin_neighbors_len():
    assert len(ORIGIN_NEIGHBORS) == 6


def test_get_neighbor_at():
    origin = ORIGIN_POSITION
    assert origin.get_neighbor_at(Direction.Up) == Position(0, -1, 0)
    assert origin.get_neighbor_at(Direction.UpRight) == Position(1, -1, 0)
    assert origin.get_neighbor_at(Direction.DownRight) == Position(1, 0, 0)
    assert origin.get_neighbor_at(Direction.Down) == Position(0, 1, 0)
    assert origin.get_neighbor_at(Direction.DownLeft) == Position(-1, 1, 0)
    assert origin.get_neighbor_at(Direction.UpLeft) == Position(-1, 0, 0)


def test_get_above():
    p = Position(2, 3, 4)
    assert p.get_above() == Position(2, 3, 5)


def test_get_below():
    p = Position(2, 3, 4)
    assert p.get_below() == Position(2, 3, 3)


def test_get_bottom():
    assert Position(1, 2, 0).get_bottom() == Position(1, 2, 0)
    assert Position(1, 2, 3).get_bottom() == Position(1, 2, 0)


def test_neighbor_deltas():
    assert NEIGHBOR_DELTAS[Direction.Up.value] == (0, -1, 0)
    assert NEIGHBOR_DELTAS[Direction.UpRight.value] == (1, -1, 0)
    assert NEIGHBOR_DELTAS[Direction.DownRight.value] == (1, 0, 0)
    assert NEIGHBOR_DELTAS[Direction.Down.value] == (0, 1, 0)
    assert NEIGHBOR_DELTAS[Direction.DownLeft.value] == (-1, 1, 0)
    assert NEIGHBOR_DELTAS[Direction.UpLeft.value] == (-1, 0, 0)
