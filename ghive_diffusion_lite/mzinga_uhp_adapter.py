"""Adapter: Mzinga's native C# engine as a move-policy callable via UHP.

``MzingaUHPAdapter`` spawns the pre-built ``MzingaEngine`` binary (from the
official `jonthysell/Mzinga` GitHub releases) as a subprocess and drives it
over the **Universal Hive Protocol** (UHP) on stdin/stdout.

This uses the *real*, fully-trained Mzinga AI (negamax + quiescence search
with transposition tables and learned metric weights) as the teacher instead
of the in-repo AlphaZero model, which had been performing poorly.

Each ``evaluate(board)`` call:

1. Syncs the engine to the Python board's position with a single
   ``newgame <GameString>`` command (the Python port's game string is
   directly accepted by UHP).
2. Runs ``bestmove depth N`` with ``ReportIntermediateBestMoves`` enabled.
   Intermediate lines have the format ``move;depth;score;PV...``; the final
   line is just the chosen move.
3. The **score** from the deepest intermediate line is the negamax evaluation
   from the side-to-move's perspective (positive = winning). It is squashed
   to ``[-1, +1]`` via ``tanh(score / value_scale)`` to serve as a value
   target. Terminal outcomes still override this via the existing backfill in
   ``gen_data.py``.

The move-string format is identical between the Python port and UHP, so the
returned ``label_move`` is directly comparable to ``legal_strs`` from the
pipeline's context builder.

Fork safety
-----------
Unlike ``MzingaMCTSAdapter`` (which preloads a torch model in the parent and
shares it via COW), this adapter must **not** be created before ``fork``.
Each worker creates its own adapter (and its own ``MzingaEngine`` subprocess)
after forking. The engine binary itself is lightweight (~12 MB, no torch).
"""

from __future__ import annotations

import math
import os
import platform
import subprocess
import threading
from typing import Optional, Tuple

from mzinga.core.board import Board


# ─── binary discovery ──────────────────────────────────────────────────────

def _candidate_binary_paths() -> list[str]:
    """Search order for the MzingaEngine binary."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates: list[str] = []
    # 1. Explicit env override
    env_path = os.environ.get("MZINGA_ENGINE_PATH")
    if env_path:
        candidates.append(env_path)
    # 2. <project>/mzinga_uhp/Mzinga.<platform>/MzingaEngine
    project_root = os.path.dirname(here)
    macos_dir = os.path.join(project_root, "mzinga_uhp")
    if os.path.isdir(macos_dir):
        for name in sorted(os.listdir(macos_dir)):
            sub = os.path.join(macos_dir, name)
            if os.path.isdir(sub):
                candidates.append(os.path.join(sub, "MzingaEngine"))
    # 3. Common system locations
    candidates.extend([
        "/usr/local/bin/MzingaEngine",
        "/opt/homebrew/bin/MzingaEngine",
        os.path.expanduser("~/bin/MzingaEngine"),
    ])
    return candidates


def find_mzinga_engine() -> str:
    """Return the path to the MzingaEngine binary, raising if not found."""
    for path in _candidate_binary_paths():
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        "MzingaEngine binary not found. Download it from "
        "https://github.com/jonthysell/Mzinga/releases/latest and either:\n"
        "  - extract it into <project>/mzinga_uhp/Mzinga.<platform>/, or\n"
        "  - set the MZINGA_ENGINE_PATH environment variable to its path."
    )


def _arch_subdir() -> str:
    """Return the expected platform subdir name for auto-download hints."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        return "Mzinga.MacOSArm64" if machine in ("arm64", "aarch64") else "Mzinga.MacOSX64"
    if system == "linux":
        return "Mzinga.LinuxArm64" if machine in ("arm64", "aarch64") else "Mzinga.LinuxX64"
    if system == "windows":
        return "Mzinga.WinArm64" if machine in ("arm64", "aarch64") else "Mzinga.WinX64"
    return "Mzinga.MacOSArm64"


# ─── UHP line parsing ──────────────────────────────────────────────────────

def _parse_intermediate_line(line: str) -> Optional[Tuple[str, int, float]]:
    """Parse a ``move;depth;score;PV...`` intermediate bestmove line.

    Returns ``(move, depth, score)`` or ``None`` if the line doesn't match.
    """
    parts = line.split(";")
    if len(parts) < 3:
        return None
    move = parts[0].strip()
    try:
        depth = int(parts[1])
        score = float(parts[2])
    except (ValueError, IndexError):
        return None
    if not move:
        return None
    return move, depth, score


def score_to_value(score: float, scale: float = 20000.0) -> float:
    """Squash a negamax score to [-1, +1] via tanh.

    The engine uses ±infinity for terminal positions (clamped to ±1e6
    internally via TreeStrapInfinity). Heuristic midgame scores are typically
    in the thousands-to-tens-of-thousands range. ``tanh(score / scale)``
    maps:
      score = 0       → 0.0   (equal)
      score = scale   → 0.76  (clear advantage)
      score = 2*scale → 0.96  (winning)
      score = ±1e6    → ±1.0  (terminal)
    """
    if math.isinf(score):
        return 1.0 if score > 0 else -1.0
    if math.isnan(score):
        return 0.0
    return math.tanh(score / scale)


# ─── adapter ───────────────────────────────────────────────────────────────

class MzingaUHPAdapter:
    """Wraps the native Mzinga C# engine as a board→move_string callable.

    Parameters
    ----------
    binary_path : str, optional
        Path to the MzingaEngine binary. Auto-discovered if omitted.
    depth : int
        Search depth for ``bestmove depth N`` (default 4). Higher = stronger
        but slower. Depth 3-5 is a good range for data generation.
    sample : bool
        Accepted for interface compatibility with ``MzingaMCTSAdapter``.
        When True, ``play_move`` uses epsilon-greedy exploration (with
        probability ``epsilon`` a random legal move is chosen) to add game
        diversity. ``label_move`` is always the engine's best move.
    epsilon : float
        Probability of playing a random legal move instead of the best move
        when ``sample=True`` (default 0.05).
    value_scale : float
        Divisor for the tanh value squash (default 20000.0).
    """

    def __init__(
        self,
        binary_path: Optional[str] = None,
        depth: int = 4,
        sample: bool = True,
        epsilon: float = 0.05,
        value_scale: float = 20000.0,
    ):
        self.binary_path = binary_path or find_mzinga_engine()
        self.depth = depth
        self.sample = sample
        self.epsilon = epsilon
        self.value_scale = value_scale

        # Spawn the engine subprocess.
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._start_engine()

        # Populated by the most recent evaluate()/__call__().
        self.last_value: float = 0.0
        self.last_label_move: Optional[str] = None
        self.last_play_move: Optional[str] = None

    def _start_engine(self) -> None:
        """Spawn the MzingaEngine subprocess and configure it."""
        self._proc = subprocess.Popen(
            [self.binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        # Read the id + capabilities banner (2 lines) then the first 'ok'.
        self._read_until_ok(timeout=10)
        # Enable intermediate bestmove reporting so we get scores.
        self._send("options set ReportIntermediateBestMoves true")
        self._read_until_ok(timeout=5)

    def _send(self, command: str) -> None:
        """Write a UHP command to the engine's stdin."""
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Engine subprocess is not running")
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()

    def _readline(self, timeout: Optional[float] = None) -> Optional[str]:
        """Read one line from the engine's stdout, stripping the trailing newline."""
        if self._proc is None or self._proc.stdout is None:
            return None
        line = self._proc.stdout.readline()
        if not line:
            return None
        return line.rstrip("\r\n")

    def _read_until_ok(self, timeout: Optional[float] = None) -> list[str]:
        """Read lines until a standalone ``ok`` line is seen.

        Returns all lines *before* the ``ok`` (excluding it). Error lines
        starting with ``err`` or ``invalidmove`` are returned in the list
        but flagged via the ``error`` attribute on the instance.
        """
        lines: list[str] = []
        while True:
            line = self._readline(timeout=timeout)
            if line is None:
                raise RuntimeError("MzingaEngine closed its stdout unexpectedly")
            stripped = line.strip()
            if stripped == "ok":
                break
            if stripped:
                lines.append(stripped)
        return lines

    def evaluate(self, board: Board) -> Tuple[Optional[str], Optional[str], float]:
        """Run one engine search; return ``(label_move, play_move, root_value)``.

        - ``label_move``: the engine's best move (supervised target).
        - ``play_move``: best move, or a random legal move with prob epsilon
          when ``self.sample`` is True.
        - ``root_value``: squashed negamax score from the side-to-move's
          perspective, in ``[-1, +1]``.
        """
        with self._lock:
            label_move, play_move, value = self._evaluate_locked(board)
        self.last_label_move = label_move
        self.last_play_move = play_move
        self.last_value = value
        return label_move, play_move, value

    def _evaluate_locked(self, board: Board) -> Tuple[Optional[str], Optional[str], float]:
        # 1. Sync the engine to the board's current position.
        game_string = board.get_game_string()
        self._send(f"newgame {game_string}")
        self._read_until_ok(timeout=10)

        # 2. Check for legal moves via the Python board (avoids an extra UHP
        #    round-trip and keeps the move format consistent).
        valid = list(board.get_valid_moves())
        if not valid:
            return "pass", "pass", 0.0

        # 3. Run the search.
        self._send(f"bestmove depth {self.depth}")
        raw_lines = self._read_until_ok(timeout=120)

        # 4. Parse: intermediate lines have 'move;depth;score;PV', the final
        #    non-ok line is just the chosen move.
        best_move: Optional[str] = None
        last_score: float = 0.0
        for line in raw_lines:
            parsed = _parse_intermediate_line(line)
            if parsed is not None:
                move, _depth, score = parsed
                best_move = move  # keep updating; last intermediate wins
                last_score = score
            else:
                # A bare move string (the final result line).
                stripped = line.strip()
                # Skip error lines.
                if stripped.startswith("err") or stripped.startswith("invalidmove"):
                    continue
                # A move string contains no ';'.
                if ";" not in stripped and stripped:
                    best_move = stripped

        if best_move is None:
            return "pass", "pass", 0.0

        # 5. Normalise the score to [-1, +1].
        value = score_to_value(last_score, self.value_scale)

        # 6. Determine play_move (epsilon-greedy if sampling).
        play_move = best_move
        if self.sample and self.epsilon > 0:
            import random as _r
            if _r.random() < self.epsilon:
                legal_strs: list[str] = []
                for m in valid:
                    try:
                        s = board.get_move_string(m)
                        if s:
                            legal_strs.append(s)
                    except Exception:
                        continue
                if legal_strs:
                    play_move = _r.choice(legal_strs)

        return best_move, play_move, value

    def __call__(self, board: Board) -> str:
        """Return the play move. Prefer ``evaluate`` for labels + value."""
        _label, play, _value = self.evaluate(board)
        if play is None:
            return "pass"
        return play

    def close(self) -> None:
        """Terminate the engine subprocess."""
        with self._lock:
            if self._proc is not None:
                try:
                    if self._proc.stdin is not None:
                        self._send("exit")
                except Exception:
                    pass
                try:
                    self._proc.wait(timeout=3)
                except Exception:
                    self._proc.kill()
                self._proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
