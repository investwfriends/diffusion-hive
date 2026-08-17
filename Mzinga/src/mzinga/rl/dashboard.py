"""Zero-dependency terminal dashboard for training — ANSI escape codes only.

Provides in-place refresh with a background clock thread, sparklines (▁▂▃▄▅▆▇█),
outcome bar charts, a liveness spinner, and a JSONL metrics logger for post-hoc
analysis.

Uses two locks: a data lock (briefly held for state reads/writes, never during
I/O) and a print lock (serialises terminal output).  The slow print() calls
never block the training loop.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from collections import deque

SPARK_CHARS = "▁▂▃▄▅▆▇█"
_SPINNER = "|/-\\"

_BOX = {
    "tl": "╔",
    "tr": "╗",
    "bl": "╚",
    "br": "╝",
    "h": "═",
    "v": "║",
    "cl": "╠",
    "cr": "╣",
    "ch": "╦",
    "cb": "╩",
}
CURSOR_UP = "\033[{}A"
CLEAR_LINE = "\033[K"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"

_DASHBOARD_HEIGHT = 13


class MetricsLogger:
    def __init__(self, path: str = "training_metrics.jsonl"):
        self.path = path
        self._fd = open(path, "w")

    def log(self, **kwargs):
        kwargs["timestamp"] = time.time()
        self._fd.write(json.dumps(kwargs) + "\n")
        self._fd.flush()

    def close(self):
        self._fd.close()


def _terminal_width():
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 80


def _sparkline(values, width=16):
    if not values:
        return " " * width
    vs = list(values)[-width:]
    vmin, vmax = min(vs), max(vs)
    span = vmax - vmin
    if span < 1e-8:
        c = SPARK_CHARS[0] if vmax < 1e-6 else SPARK_CHARS[-1]
        return c * len(vs)
    return "".join(
        SPARK_CHARS[
            min(len(SPARK_CHARS) - 1, int((v - vmin) / span * (len(SPARK_CHARS) - 1)))
        ]
        for v in vs
    )


def _progress_bar(current, total, width=20):
    frac = min(1.0, current / total) if total > 0 else 0.0
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)


def _format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _bar_chart(value, max_val, width):
    if max_val == 0:
        return "░" * width
    frac = min(1.0, max(0.0, value / max_val))
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)


class TerminalDashboard:
    def __init__(
        self,
        n_iterations: int,
        title: str = "Training",
        log_path: str = "training_metrics.jsonl",
        spark_width: int = 16,
    ):
        self.n_iterations = n_iterations
        self.title = title
        self.spark_width = spark_width
        self.logger = MetricsLogger(log_path)
        self._histories: dict[str, deque] = {}
        self._last_kwargs: dict = {}
        self._last_iteration = 0
        self._start_time = time.time()
        self._started = False
        self._lock = threading.Lock()
        self._print_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._update_received = threading.Event()
        self._throughput = 0.0
        self._last_update_time = self._start_time
        self._spinner_idx = 0

        for _ in range(_DASHBOARD_HEIGHT):
            print()
        print(HIDE_CURSOR, end="", flush=True)

        self._clock_thread = threading.Thread(target=self._clock_ticker, daemon=True)
        self._clock_thread.start()

    # ------------------------------------------------------------------
    #  public API
    # ------------------------------------------------------------------

    def track(self, name: str, value: float):
        with self._lock:
            if name not in self._histories:
                self._histories[name] = deque(maxlen=100)
            self._histories[name].append(value)

    def update(self, iteration: int, win_rate: float = 0.0, **kwargs):
        now = time.time()
        dt = now - self._last_update_time
        if dt > 0:
            instant = 1.0 / dt
            alpha = 0.3
            self._throughput = (1.0 - alpha) * self._throughput + alpha * instant
        self._last_update_time = now

        new_kwargs = dict(
            iteration=iteration,
            win_rate=win_rate,
            policy_loss=kwargs.get("policy_loss", 0.0),
            value_loss=kwargs.get("value_loss", 0.0),
            grad_norm=kwargs.get("grad_norm", 0.0),
            policy_entropy=kwargs.get("policy_entropy"),
            mcts_entropy=kwargs.get("mcts_entropy"),
            buf_size=kwargs.get("buf_size", 0),
            lr=kwargs.get("lr", 0.0),
            avg_moves=kwargs.get("avg_moves", 0.0),
            terminal_rate=kwargs.get("terminal_rate", 0.0),
            total_games=kwargs.get("total_games", 0),
            eval_wins=kwargs.get("eval_wins"),
            eval_losses=kwargs.get("eval_losses"),
            eval_draws=kwargs.get("eval_draws"),
        )

        with self._lock:
            self._last_iteration = iteration
            self._last_kwargs = new_kwargs
            self._hist_track_locked("p_loss", new_kwargs["policy_loss"])
            self._hist_track_locked("v_loss", new_kwargs["value_loss"])
            self._hist_track_locked("win_rate", win_rate)
            if new_kwargs["grad_norm"] is not None:
                self._hist_track_locked("grad_norm", new_kwargs["grad_norm"])
            if new_kwargs["policy_entropy"] is not None:
                self._hist_track_locked("policy_entropy", new_kwargs["policy_entropy"])
            if new_kwargs["mcts_entropy"] is not None:
                self._hist_track_locked("mcts_entropy", new_kwargs["mcts_entropy"])
            self._started = True

        # render happens *outside* the data lock — no I/O while locked
        self._render_to_screen()

        self._update_received.set()

        elapsed = time.time() - self._start_time
        eta = (
            elapsed / max(1, iteration) * (self.n_iterations - iteration)
            if iteration > 0
            else 0
        )
        self.logger.log(
            iteration=iteration,
            policy_loss=new_kwargs["policy_loss"],
            value_loss=new_kwargs["value_loss"],
            win_rate=win_rate,
            buffer_size=new_kwargs["buf_size"],
            lr=new_kwargs["lr"],
            avg_moves=new_kwargs["avg_moves"],
            terminal_rate=new_kwargs["terminal_rate"],
            total_games=new_kwargs["total_games"],
            grad_norm=new_kwargs["grad_norm"],
            policy_entropy=new_kwargs["policy_entropy"],
            mcts_entropy=new_kwargs["mcts_entropy"],
            throughput=self._throughput,
            elapsed=elapsed,
            eta=eta,
        )

    def close(self):
        self._stop_event.set()
        if self._clock_thread.is_alive():
            self._clock_thread.join(timeout=2.0)
        self.logger.close()
        with self._print_lock:
            print(SHOW_CURSOR, end="", flush=True)

    # ------------------------------------------------------------------
    #  internal helpers
    # ------------------------------------------------------------------

    def _hist_track_locked(self, name: str, value: float):
        """Caller must hold self._lock."""
        if name not in self._histories:
            self._histories[name] = deque(maxlen=100)
        self._histories[name].append(value)

    def _clock_ticker(self):
        while not self._stop_event.is_set():
            if self._stop_event.wait(1.0):
                break
            if not self._update_received.is_set():
                continue
            self._spinner_idx = (self._spinner_idx + 1) % len(_SPINNER)
            self._render_to_screen()

    # ------------------------------------------------------------------
    #  rendering — pure line-building, separated from I/O
    # ------------------------------------------------------------------

    def _render_to_screen(self):
        # 1. snapshot mutable state under brief data lock
        with self._lock:
            histories = {k: list(v) for k, v in self._histories.items()}
            kw = dict(self._last_kwargs)
            iteration = self._last_iteration
            started = self._started
            throughput = self._throughput
            last_update = time.time() - self._last_update_time
            spinner = _SPINNER[self._spinner_idx]

        # 2. pure computation — zero locks, zero I/O
        lines = _build_lines(
            title=self.title,
            n_iterations=self.n_iterations,
            spark_width=self.spark_width,
            histories=histories,
            kw=kw,
            iteration=iteration,
            throughput=throughput,
            spinner=spinner,
            start_time=self._start_time,
            last_update_s=last_update,
        )

        # 3. serialised terminal I/O
        with self._print_lock:
            if started:
                print(CURSOR_UP.format(_DASHBOARD_HEIGHT), end="")
            for line in lines:
                print(f"{CLEAR_LINE}{line}")
            print(flush=True)


# ======================================================================
#  pure function — callable from *any* thread without touching Dashboard
# ======================================================================

def _build_lines(
    *,
    title: str,
    n_iterations: int,
    spark_width: int,
    histories: dict[str, list[float]],
    kw: dict,
    iteration: int,
    throughput: float,
    spinner: str,
    start_time: float,
    last_update_s: float,
) -> list[str]:

    elapsed = time.time() - start_time
    eta = (
        elapsed / max(1, iteration) * (n_iterations - iteration)
        if iteration > 0
        else 0
    )

    policy_loss = kw.get("policy_loss", 0.0)
    value_loss = kw.get("value_loss", 0.0)
    grad_norm = kw.get("grad_norm", 0.0)
    policy_entropy = kw.get("policy_entropy")
    mcts_entropy = kw.get("mcts_entropy")
    win_rate = kw.get("win_rate", 0.0)
    buf_size = kw.get("buf_size", 0)
    lr = kw.get("lr", 0.0)
    avg_moves = kw.get("avg_moves", 0.0)
    terminal_rate = kw.get("terminal_rate", 0.0)
    total_games = kw.get("total_games", 0)
    eval_wins = kw.get("eval_wins")
    eval_losses = kw.get("eval_losses")
    eval_draws = kw.get("eval_draws")

    w = min(100, _terminal_width())
    inner = max(40, w - 2)
    sw = spark_width
    half = inner // 2
    lines: list[str] = []
    bh = _BOX["h"] * inner

    def _pad(s, n):
        return s + " " * max(0, n - len(s))

    def _h(name: str) -> list[float]:
        return histories.get(name, [])

    def _metric_col(label, name, value, fmt, fallback=None):
        spark = _sparkline(_h(name), sw)
        if value is not None:
            return f"{label} {spark} {value:{fmt}}"
        if fallback is not None:
            return _metric_col(label, name, fallback, fmt)
        hist = _h(name)
        if hist:
            return f"{label} {spark}    ---"
        return f"{' ' * (len(label) + 1 + sw + 8)}"

    # — title —
    lines.append(f"{_BOX['tl']}{bh}{_BOX['tr']}")
    title_text = f" {title} "
    lines.append(f"{_BOX['v']}{_pad(title_text, inner)}{_BOX['v']}")
    lines.append(f"{_BOX['cl']}{bh}{_BOX['cr']}")

    # — progress bar + spinner + stall warn —
    bar = _progress_bar(iteration, n_iterations, min(30, inner - 30))
    tp = f"  {throughput:.1f} it/s" if throughput > 0 else ""
    stall = f"  last: {_format_time(last_update_s)}" if last_update_s >= 1.0 else ""
    progress_text = f" {bar}  {iteration}/{n_iterations}  ETA {_format_time(eta)}{tp}{stall}  {spinner}"
    lines.append(f"{_BOX['v']}{_pad(progress_text, inner)}{_BOX['v']}")
    lines.append(f"{_BOX['cl']}{bh}{_BOX['cr']}")

    # — LOSSES | EXPLORATION split —
    header_l = " LOSSES"
    header_r = " EXPLORATION"
    lines.append(
        f"{_BOX['v']} {_pad(header_l, half)}{header_r}"
        f"{' ' * (inner - half - len(header_r))}{_BOX['v']}"
    )

    p_line = _metric_col("p_loss", "p_loss", policy_loss, ".4f")
    pe_val = policy_entropy
    pe_line = _metric_col("π_ent ", "policy_entropy", pe_val, ".3f", fallback=0.0).strip()
    comb1 = f" {_pad(p_line, half)}{pe_line} "
    lines.append(f"{_BOX['v']}{_pad(comb1, inner)}{_BOX['v']}")

    v_line = _metric_col("v_loss", "v_loss", value_loss, ".4f")
    me_line = _metric_col("mcts  ", "mcts_entropy", mcts_entropy, ".3f", fallback=0.0).strip()
    comb2 = f" {_pad(v_line, half)}{me_line} "
    lines.append(f"{_BOX['v']}{_pad(comb2, inner)}{_BOX['v']}")

    g_line = _metric_col("g_norm", "grad_norm", grad_norm, ".4f")
    lines.append(f"{_BOX['v']} {_pad(g_line, inner)}{_BOX['v']}")

    lines.append(f"{_BOX['cl']}{bh}{_BOX['cr']}")

    # — eval section —
    wr_spark = _sparkline(_h("win_rate"), sw // 2)
    line_prefix = f" win% {wr_spark} {win_rate:5.2f}"
    if eval_wins is not None and eval_losses is not None and eval_draws is not None:
        total_ev = eval_wins + eval_losses + eval_draws
        bar_w = 8
        w_bar = _bar_chart(eval_wins, max(1, total_ev), bar_w)
        l_bar = _bar_chart(eval_losses, max(1, total_ev), bar_w)
        d_bar = _bar_chart(eval_draws, max(1, total_ev), bar_w)
        eval_chunk = f"  W{w_bar} L{l_bar} D{d_bar}"
    else:
        eval_chunk = ""
    line_prefix += eval_chunk
    lines.append(f"{_BOX['v']} {_pad(line_prefix, inner)}{_BOX['v']}")

    # — stats —
    stats = (
        f"buf {buf_size:>7,}  lr {lr:.2e}  games {total_games}"
        f"  moves {avg_moves:.0f}  term {terminal_rate:.0%}"
    )
    lines.append(f"{_BOX['v']} {_pad(stats, inner)}{_BOX['v']}")

    # — elapsed —
    elapsed_line = f"elapsed: {_format_time(elapsed)}"
    lines.append(f"{_BOX['v']} {_pad(elapsed_line, inner)}{_BOX['v']}")

    lines.append(f"{_BOX['bl']}{bh}{_BOX['br']}")
    return lines
