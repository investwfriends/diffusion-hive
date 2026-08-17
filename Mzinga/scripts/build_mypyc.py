"""Compile Mzinga's hot-path modules with mypyc for faster self-play.

Run from project root:
    uv run python scripts/build_mypyc.py

This compiles the Board engine (mzinga.core.{position, move, fast_set, board})
to native code via mypyc. The result: perft runs ~2.7x faster, MCTS self-play
~6% faster.

We deliberately leave mzinga.rl.mcts interpreted: profiling showed mypyc on
the MCTS actually slows it down (the bottleneck inside MCTS is dict operations
and numpy interop, not Python loop overhead — mypyc doesn't help there and
the type-checking overhead it adds is a net loss).

The resulting .so files are dropped next to the .py source files in
src/mzinga/core/ and shadow the .py on import. Delete the .so files
(rm src/mzinga/core/*.so) to fall back to interpreted Python.

Requires the `mypyc` extra (mypyc is bundled with `mypy`, installed
automatically by `uv sync`).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"

# These are the modules that mypyc actually makes faster. MCTS is
# deliberately excluded (see module docstring for rationale).
HOT_MODULES = [
    SRC / "mzinga" / "core" / "position.py",
    SRC / "mzinga" / "core" / "move.py",
    SRC / "mzinga" / "core" / "fast_set.py",
    SRC / "mzinga" / "core" / "board.py",
]


def clean_previous() -> None:
    for d in [SRC, SRC / "mzinga" / "core", SRC / "mzinga" / "rl"]:
        for so in d.glob("*.so"):
            so.unlink()
    for d in [SRC, SRC / "mzinga" / "core", SRC / "mzinga" / "rl",
              SRC / "mzinga" / "gym", SRC / "mzinga" / "rl",
              SRC / "mzinga" / "mzinga_perft", SRC / "tests"]:
        pycache = d / "__pycache__"
        if pycache.exists():
            shutil.rmtree(pycache, ignore_errors=True)
    build = PROJECT_ROOT / "build"
    if build.exists():
        shutil.rmtree(build)


def compile_all() -> int:
    clean_previous()
    cmd = [sys.executable, "-m", "mypyc", *(str(p) for p in HOT_MODULES)]
    print(f"Compiling with mypyc: {', '.join(p.name for p in HOT_MODULES)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def main() -> int:
    rc = compile_all()
    if rc == 0:
        print()
        print("Done. Verify with:")
        print("  uv run pytest                 # all tests should pass")
        print("  uv run mzinga-perft 5 Base    # should be ~1000 KN/s")
        print("  uv run python scripts/bench_mcts_long.py 2 200  # ~900 sims/s")
        print()
        print("Revert with: rm src/mzinga/core/*.so")
    return rc


if __name__ == "__main__":
    sys.exit(main())
