import asyncio
import sys
import time

from mzinga.core.board import Board
from mzinga.core.enums import GameType, try_parse_game_type


def format_number(n: int) -> str:
    return f"{n:>18}"


def main():
    args = sys.argv[1:]

    max_depth = 2**31 - 1
    game_type = GameType.Base
    game_string = None
    use_mt = False

    for arg in args:
        if arg == "-mt":
            use_mt = True
        elif arg.isdigit() or (arg.startswith("-") and arg[1:].isdigit()):
            max_depth = int(arg)
        elif ";" in arg:
            game_string = arg
        else:
            parsed = try_parse_game_type(arg)
            if parsed is not None:
                game_type = parsed
            elif not arg.startswith("-"):
                game_string = arg

    if game_string:
        board = Board.parse_game_string(game_string)
    else:
        board = Board(game_type)

    print("MzingaPerft")
    print()

    for depth in range(max_depth + 1):
        start_time = time.time()

        if use_mt:
            nodes = board.parallel_perft(depth)
        else:
            # Use the synchronous implementation directly to skip asyncio frame
            # overhead at the recursion depths used by perft. Public API is
            # preserved: tests/test_perft.py still calls calculate_perft_async.
            nodes = board._calculate_perft_sync(depth)

        elapsed_ms = int((time.time() - start_time) * 1000)
        kn_s = (nodes / max(elapsed_ms, 1)) if elapsed_ms > 0 else 0.0

        print(f"perft({depth:2d})   = {format_number(nodes)} in {elapsed_ms:6d} ms. {kn_s:8.1f} KN/s")


if __name__ == "__main__":
    main()
