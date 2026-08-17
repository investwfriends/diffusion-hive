#!/usr/bin/env python3
"""Parallel game generation using Mzinga's AlphaZero + MCTS.

Architecture (startup-memory safe)
----------------------------------
1. Parent imports torch and loads the teacher model **once**.
2. Workers are started with ``fork`` so they inherit the model via COW
   (read-only weights stay shared; each process privately owns its MCTS
   tree after the first write).
3. Optional light stagger spreads first-touch COW faults so peak RSS
   stays well below total RAM.
4. Progress is file-based (no Queue/Manager) to avoid deadlocks.

Usage::

    PYTHONPATH="..." python -m ghive_diffusion_lite.gen_data \\
        --games 400 --workers 5 --output dataset.pt
"""

from __future__ import annotations

import os
# MUST be set before importing torch — prevents fork+OpenMP deadlock
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import multiprocessing as mp
import random
import shutil
import tempfile
import time

import torch
from mzinga.core.enums import GameType

from ghive_diffusion.context_builder import HiveContextBuilder
from ghive_diffusion.dataset import game_outcome_value, make_random_policy
from ghive_diffusion.tokenizer import build_default_tokenizer
from ghive_diffusion.training import compute_aux_targets

from .hive_lite_config import HiveLiteConfig
from .lite_trainer import LiteTrainingSample
from .mzinga_adapter import MzingaMCTSAdapter
from .mzinga_uhp_adapter import MzingaUHPAdapter

# Populated in the parent before fork; inherited by children under fork.
# Only used for the "alphazero" teacher (torch model shared via COW).
# The "uhp" teacher spawns a subprocess per worker and must NOT be preloaded.
_SHARED_ADAPTER: MzingaMCTSAdapter | None = None
_SHARED_TEACHER: str = "uhp"  # "uhp" (native C# engine) or "alphazero"
_SHARED_UHP_DEPTH: int = 4
_SHARED_UHP_EPSILON: float = 0.05
_SHARED_SIMULATIONS: int = 50
_SHARED_SAMPLE: bool = True
_SHARED_OPPONENT: str = "random"
# Blend mode: fraction of games played vs random (rest are teacher self-play).
# When not None, overrides _SHARED_OPPONENT on a per-game basis.
_SHARED_RANDOM_FRACTION: float | None = None
_SHARED_COUNTERFACTUAL: bool = True
_SHARED_SCRAMBLE_PLIES: int = 0
# Context size controls. ``history_window`` caps how many trailing move
# strings go into the <history> section. The full transcript pushes
# contexts to ~2100 tokens by the midgame, and attention is O(T^2), so
# an unbounded window dominates training cost. None = full transcript.
_SHARED_HISTORY_WINDOW: int | None = None
# Hard ply cap per game. Games that hit it are unfinished, so they get
# no terminal outcome backfill and contribute no value signal.
_SHARED_MAX_PLIES: int = 400
# Drop samples from games that never reached a terminal state. Keeps the
# value head from training on a majority of value=0 (teacher-root-only)
# labels at the cost of throwing away the unfinished games.
_SHARED_DROP_UNFINISHED: bool = False
# Streaming shard settings (inherited by fork workers).
_SHARD_BYTES: int = 0  # 0 = disabled; else rotate when buffer ~this many bytes
_SHARD_DIR: str | None = None
# Empirical mean from recent MCTS shard dumps (~15.2 KB / sample on disk).
_BYTES_PER_SAMPLE_EST: int = 15200


def _available_ram_gb() -> float | None:
    """Best-effort free/available RAM in GiB (Linux /proc, else None)."""
    try:
        with open("/proc/meminfo") as f:
            total = available = None
            for line in f:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) / (1024 * 1024)
                elif line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / (1024 * 1024)
            return available if available is not None else total
    except OSError:
        return None


def recommend_workers(
    ram_gb: float | None = None,
    cpus: int | None = None,
) -> int:
    """Recommend worker count: min(CPU headroom, RAM headroom).

    With parent-preload + fork COW, steady-state is roughly:
      shared torch/model ≈ 0.2 GB
      private per worker  ≈ 0.4–0.6 GB (Python heap + samples + MCTS)
    We budget 0.6 GB/worker and leave 3 GB for OS/parent/merge peak.
    """
    if cpus is None:
        cpus = os.cpu_count() or 4
    if ram_gb is None:
        ram_gb = _available_ram_gb()
    if ram_gb is None:
        ram_gb = 16.0  # conservative unknown

    # Leave one logical CPU for the parent / OS bookkeeping.
    cpu_cap = max(1, cpus - 1)
    # RAM budget after OS reserve.
    mem_cap = max(1, int((ram_gb - 3.0) / 0.6))
    return max(1, min(cpu_cap, mem_cap))


def _flush_shard(
    samples: list[LiteTrainingSample],
    worker_id: int,
    shard_idx: int,
    shard_dir: str,
) -> tuple[int, int]:
    """Write a finalized shard; return (new_shard_idx, bytes_written)."""
    if not samples:
        return shard_idx, 0
    path = os.path.join(
        shard_dir, f"shard_w{worker_id:03d}_{shard_idx:05d}.pt"
    )
    # Write to temp then rename so pullers never see a partial file.
    tmp = path + ".tmp"
    torch.save(samples, tmp)
    os.replace(tmp, path)
    nbytes = os.path.getsize(path)
    return shard_idx + 1, nbytes


# Option 4: number of future plies to denoise as a "plan". The diffusion
# canvas target becomes the next _PLAN_LEN moves of the actual game (each move
# already ends in an EOS token, so the plan is a self-delimiting sequence).
_PLAN_LEN = 4


def _worker(n_games: int, worker_id: int, seed: int, output_dir: str) -> None:
    """Generate *n_games* games, save to output_dir/worker_{id}.pt."""
    try:
        random.seed(seed + worker_id * 1000)
        torch.manual_seed(seed + worker_id * 1000)
        torch.set_num_threads(1)

        # Build the teacher adapter.
        # - "alphazero": use the parent-preloaded adapter (fork COW) if
        #   available, else load privately.
        # - "uhp": always create a fresh subprocess per worker (the native
        #   C# engine cannot be shared across forked processes).
        if _SHARED_TEACHER == "uhp":
            mz = MzingaUHPAdapter(
                depth=_SHARED_UHP_DEPTH,
                sample=_SHARED_SAMPLE,
                epsilon=_SHARED_UHP_EPSILON,
            )
        elif _SHARED_ADAPTER is not None:
            mz = _SHARED_ADAPTER
        else:
            mz = MzingaMCTSAdapter(
                device="cpu", num_simulations=_SHARED_SIMULATIONS, sample=_SHARED_SAMPLE
            )

        tk = build_default_tokenizer(HiveLiteConfig())
        builder = HiveContextBuilder(tk, history_window=_SHARED_HISTORY_WINDOW)

        from mzinga.core.board import Board
        from mzinga.core.enums import GameType, PlayerColor
        rand_policy = make_random_policy()

        samples: list[LiteTrainingSample] = []
        prog_file = os.path.join(output_dir, f"progress_{worker_id}.txt")
        save_file = os.path.join(output_dir, f"worker_{worker_id}.pt")
        stop_file = os.path.join(output_dir, "STOP")
        shard_idx = 0
        shard_bytes_written = 0
        bytes_file = os.path.join(output_dir, f"bytes_{worker_id}.txt")
        # Decisiveness bookkeeping — surfaced in the run summary so a
        # draw-heavy config is visible immediately rather than after a
        # training run produces a dead value head.
        stats_file = os.path.join(output_dir, f"stats_{worker_id}.txt")
        games_finished = 0
        games_unfinished = 0

        for gi in range(n_games):
            if os.path.exists(stop_file):
                break

            board = Board(GameType.Base)
            teacher_is_white = (gi % 2 == 0)

            # Per-game opponent selection.
            # - Blend mode (_SHARED_RANDOM_FRACTION is not None): each game
            #   independently plays vs random with that probability, else
            #   teacher self-play (teacher plays both sides).
            # - Legacy mode: honours _SHARED_OPPONENT ("random" or "teacher").
            if _SHARED_RANDOM_FRACTION is not None:
                game_vs_random = random.random() < _SHARED_RANDOM_FRACTION
            else:
                game_vs_random = (_SHARED_OPPONENT == "random")

            # 1. Scramble opening plies if requested
            for _ in range(_SHARED_SCRAMBLE_PLIES):
                if board.game_is_over:
                    break
                moves = list(board.get_valid_moves())
                if not moves:
                    break
                mv = random.choice(moves)
                try:
                    board.trusted_play(mv, board.get_move_string(mv))
                except Exception:
                    break

            # 2. Play game — collect per-game samples so we can backfill
            # terminal outcomes from the true side-to-move at each ply
            # (scramble can make the first sample Black to move).
            game_samples: list[LiteTrainingSample] = []
            game_sides: list = []
            game_moves: list[str] = []
            game_ply_indices: list[int] = []

            for _ply in range(_SHARED_MAX_PLIES):
                if board.game_is_over:
                    break

                is_teacher_turn = (
                    (board.current_color == PlayerColor.White) == teacher_is_white
                )
                legal_strs = builder._legal_moves(board)
                if not legal_strs:
                    break

                side = board.current_color
                # Query the teacher for a label when:
                #  - counterfactual labelling is on (label every ply), or
                #  - it's the teacher's turn, or
                #  - the opponent is also the teacher (self-play → both sides).
                need_teacher = (
                    _SHARED_COUNTERFACTUAL
                    or is_teacher_turn
                    or not game_vs_random
                )

                label_str: str | None = None
                play_str: str | None = None
                teacher_val = 0.0

                # One MCTS search: argmax label + (optional) sampled play + root value.
                if need_teacher:
                    label_str, play_str, teacher_val = mz.evaluate(board)

                    # Only keep samples whose expert label is truly legal.
                    # Never fall back to target_legal_idx=0 (silent policy poison).
                    if label_str is not None and label_str in legal_strs:
                        target_idx = legal_strs.index(label_str)
                        ctx_ids = builder.encode(board, target_move=None)
                        legal_ids = [tk.encode_move(s) for s in legal_strs]
                        target_ids = tk.encode_move(label_str)
                        try:
                            aux = compute_aux_targets(board)
                        except Exception:
                            aux = None

                        game_samples.append(LiteTrainingSample(
                            context_ids=torch.tensor(ctx_ids, dtype=torch.long),
                            target_move_ids=torch.tensor(target_ids, dtype=torch.long),
                            legal_move_ids=legal_ids,
                            target_legal_idx=target_idx,
                            value=float(teacher_val),
                            aux_targets=aux,
                        ))
                        game_sides.append(side)
                        game_ply_indices.append(_ply)

                # Advance the board: random opponent on its turn, else teacher.
                if game_vs_random and not is_teacher_turn:
                    move_str = rand_policy(board)
                else:
                    move_str = play_str

                if move_str is None or move_str not in legal_strs:
                    move_str = random.choice(legal_strs)

                played = False
                for mv in board.get_valid_moves():
                    try:
                        if board.get_move_string(mv) == move_str:
                            board.trusted_play(mv, move_str)
                            played = True
                            break
                    except Exception:
                        continue
                if not played:
                    break
                game_moves.append(move_str)

            # Terminal outcome backfill: strongest value signal for finished games.
            # Uses the side that was to move when the sample was recorded.
            if board.game_is_over and game_samples:
                for sample, ply_side in zip(game_samples, game_sides):
                    sample.value = float(game_outcome_value(board, ply_side))
                    sample.has_outcome = True
                games_finished += 1
            elif game_samples:
                # Hit the ply cap. These samples keep the teacher's root value
                # (weak signal) and has_outcome stays False.
                games_unfinished += 1
                if _SHARED_DROP_UNFINISHED:
                    game_samples = []

            # Option 4: widen the diffusion target to a multi-ply plan.
            for si, sample in enumerate(game_samples):
                ply = game_ply_indices[si]
                plan = game_moves[ply:ply + _PLAN_LEN]
                if plan:
                    plan_ids: list[int] = []
                    for mv in plan:
                        plan_ids.extend(tk.encode_move(mv))
                    sample.target_move_ids = torch.tensor(plan_ids, dtype=torch.long)

            samples.extend(game_samples)

            with open(prog_file, "w") as f:
                f.write(str(gi + 1))

            # Rotate streaming shards by estimated on-disk size.
            if (
                _SHARD_BYTES > 0
                and _SHARD_DIR is not None
                and len(samples) * _BYTES_PER_SAMPLE_EST >= _SHARD_BYTES
            ):
                shard_idx, nbytes = _flush_shard(
                    samples, worker_id, shard_idx, _SHARD_DIR
                )
                shard_bytes_written += nbytes
                samples = []
                with open(bytes_file, "w") as f:
                    f.write(str(shard_bytes_written))

            if (gi + 1) % 10 == 0 and samples:
                torch.save(samples, save_file)

        # Final residual buffer → last shard (if streaming) and/or worker file.
        if samples:
            if _SHARD_BYTES > 0 and _SHARD_DIR is not None:
                shard_idx, nbytes = _flush_shard(
                    samples, worker_id, shard_idx, _SHARD_DIR
                )
                shard_bytes_written += nbytes
                with open(bytes_file, "w") as f:
                    f.write(str(shard_bytes_written))
            torch.save(samples, save_file)
        elif os.path.exists(save_file) is False:
            torch.save([], save_file)

        with open(stats_file, "w") as f:
            f.write(f"{games_finished} {games_unfinished}")
    except Exception as e:
        import traceback
        err_file = os.path.join(output_dir, f"error_{worker_id}.txt")
        with open(err_file, "w") as f:
            f.write(f"WORKER {worker_id} CRASHED: {e}\n{traceback.format_exc()}")
    finally:
        # Clean up the UHP subprocess if this worker created one.
        if _SHARED_TEACHER == "uhp" and "mz" in dir() and hasattr(mz, "close"):
            try:
                mz.close()
            except Exception:
                pass


def _fmt_sec(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s // 60:.0f}m {s % 60:.0f}s"
    return f"{s // 3600}h {(s % 3600) // 60:.0f}m"


def _read_progress(tmp_dir: str, worker_id: int) -> int:
    path = os.path.join(tmp_dir, f"progress_{worker_id}.txt")
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0


def _read_error(tmp_dir: str, worker_id: int) -> str | None:
    path = os.path.join(tmp_dir, f"error_{worker_id}.txt")
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return None


def _shard_dir_bytes(shard_dir: str | None) -> int:
    if not shard_dir or not os.path.isdir(shard_dir):
        return 0
    total = 0
    for name in os.listdir(shard_dir):
        if name.endswith(".pt") and not name.endswith(".tmp"):
            try:
                total += os.path.getsize(os.path.join(shard_dir, name))
            except OSError:
                pass
    return total


def _fmt_bytes(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def generate_dataset_parallel(
    n_games: int = 1000,
    num_workers: int | None = None,
    seed: int = 42,
    output: str = "dataset.pt",
    report_interval: int = 300,
    num_simulations: int = 50,
    stagger_seconds: float = 1.0,
    shard_bytes: int = 0,
    shard_dir: str | None = None,
    target_bytes: int = 0,
    merge_final: bool = True,
    sample: bool = True,
    opponent: str = "random",
    counterfactual: bool = True,
    scramble_plies: int = 6,
    teacher: str = "uhp",
    uhp_depth: int = 4,
    uhp_epsilon: float = 0.05,
    random_fraction: float | None = 0.5,
    history_window: int | None = None,
    max_plies: int = 400,
    drop_unfinished: bool = False,
):
    """Generate *n_games* teacher self-play games in parallel via fork.

    ``teacher`` selects the move-policy source:
      - ``"uhp"``      — native Mzinga C# engine via UHP subprocess (default,
                         strong, no torch needed for the teacher).
      - ``"alphazero"`` — in-repo AlphaZero model + MCTS (legacy).

    ``random_fraction`` controls the opponent blend:
      - ``None``  — honour ``opponent`` ("random" or "teacher") for all games.
      - ``float`` in [0, 1] — each game independently plays vs random with
        that probability, else teacher self-play. Default 0.5 so the model
        learns both to punish random play and to play strong positions.

    ``history_window`` caps the <history> section to the last N move
    strings. ``None`` keeps the full transcript, which grows contexts to
    ~2100 tokens by the midgame; since attention is O(T^2) this is the
    single largest driver of training cost. Contexts are stored
    pre-tokenized, so this is fixed at generation time.

    ``max_plies`` / ``drop_unfinished`` control decisiveness. Games that
    hit the ply cap get no terminal outcome backfill, so their samples
    carry only the teacher's weak root value.
    """
    global _SHARED_ADAPTER, _SHARED_TEACHER, _SHARED_UHP_DEPTH, _SHARED_UHP_EPSILON, _SHARED_SIMULATIONS, _SHARED_SAMPLE, _SHARED_OPPONENT, _SHARED_RANDOM_FRACTION, _SHARED_COUNTERFACTUAL, _SHARED_SCRAMBLE_PLIES, _SHARED_HISTORY_WINDOW, _SHARED_MAX_PLIES, _SHARED_DROP_UNFINISHED, _SHARD_BYTES, _SHARD_DIR

    if num_workers is None:
        num_workers = recommend_workers()

    tmp_dir = tempfile.mkdtemp(prefix="ghive_gen_")
    num_workers = min(num_workers, n_games)
    ram = _available_ram_gb()
    cpus = os.cpu_count() or 4
    rec = recommend_workers(ram_gb=ram, cpus=cpus)

    if shard_bytes > 0:
        if not shard_dir:
            shard_dir = os.path.join(os.path.dirname(os.path.abspath(output)) or ".", "shards")
        os.makedirs(shard_dir, exist_ok=True)
        _SHARD_BYTES = shard_bytes
        _SHARD_DIR = shard_dir
    else:
        _SHARD_BYTES = 0
        _SHARD_DIR = None

    _SHARED_TEACHER = teacher
    _SHARED_UHP_DEPTH = uhp_depth
    _SHARED_UHP_EPSILON = uhp_epsilon
    _SHARED_SIMULATIONS = num_simulations
    _SHARED_SAMPLE = sample
    _SHARED_OPPONENT = opponent
    _SHARED_RANDOM_FRACTION = random_fraction
    _SHARED_COUNTERFACTUAL = counterfactual
    _SHARED_SCRAMBLE_PLIES = scramble_plies
    _SHARED_HISTORY_WINDOW = history_window
    _SHARED_MAX_PLIES = max_plies
    _SHARED_DROP_UNFINISHED = drop_unfinished

    print(f"gen_data: {n_games} games, {num_workers} workers")
    print(f"  history window: {history_window if history_window is not None else 'full transcript'}")
    print(f"  max plies:      {max_plies}  (drop unfinished: {drop_unfinished})")
    print(f"  CPUs:           {cpus}  (recommended workers: {rec})")
    if ram is not None:
        print(f"  RAM avail:      {ram:.1f} GB")
    print(f"  teacher:        {teacher}")
    if teacher == "uhp":
        print(f"  uhp depth:      {uhp_depth}")
        print(f"  uhp epsilon:    {uhp_epsilon}")
    else:
        print(f"  sims:           {num_simulations} MCTS / move")
    print(f"  stagger:        {stagger_seconds}s between worker starts")
    if random_fraction is not None:
        print(f"  opponent:       blend ({random_fraction:.0%} vs random, "
              f"{1 - random_fraction:.0%} self-play)")
    else:
        print(f"  opponent:       {opponent}")
    print(f"  counterfactual: {counterfactual}")
    print(f"  scramble_plies: {scramble_plies}")
    print(f"  temp dir:       {tmp_dir}")
    print(f"  output:         {output}")
    if shard_bytes > 0:
        print(f"  shards:         every {_fmt_bytes(shard_bytes)} → {shard_dir}")
    if target_bytes > 0:
        print(f"  target:         {_fmt_bytes(target_bytes)} (STOP when reached)")
    print(f"  progress:       every {report_interval}s", flush=True)

    if teacher == "alphazero":
        # Load the torch model ONCE in the parent so fork children share
        # weights via copy-on-write.
        print("\nLoading AlphaZero teacher model once in parent (shared via fork)...", flush=True)
        t_load = time.time()
        _SHARED_ADAPTER = MzingaMCTSAdapter(
            device="cpu", num_simulations=num_simulations, sample=sample
        )
        print(f"  teacher ready in {_fmt_sec(time.time() - t_load)}\n", flush=True)
    else:
        # UHP teacher: each worker spawns its own MzingaEngine subprocess.
        # No parent preload — the subprocess cannot be shared across fork.
        _SHARED_ADAPTER = None
        from .mzinga_uhp_adapter import find_mzinga_engine
        try:
            binary = find_mzinga_engine()
            print(f"\nUHP teacher: MzingaEngine binary at {binary}", flush=True)
            print(f"  each worker will spawn its own engine subprocess\n", flush=True)
        except FileNotFoundError as e:
            print(f"\n  *** {e}", flush=True)
            raise

    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    start_method = mp.get_start_method(allow_none=True) or "fork"
    if start_method != "fork":
        print(
            f"  WARNING: start method is {start_method!r}, not fork. "
            "Each worker will re-load torch/model (higher RAM).",
            flush=True,
        )
        # Avoid sharing a possibly non-picklable live adapter under spawn.
        _SHARED_ADAPTER = None

    actual_workers = num_workers
    games_per_worker = n_games // actual_workers
    remainder = n_games % actual_workers

    workers: list[tuple[int, mp.Process]] = []
    for i in range(actual_workers):
        g = games_per_worker + (1 if i < remainder else 0)
        if g == 0:
            continue
        p = mp.Process(target=_worker, args=(g, i, seed, tmp_dir))
        workers.append((i, p))
        p.start()
        print(f"  Worker {i} started — {g} games", flush=True)
        if stagger_seconds > 0 and i < actual_workers - 1:
            time.sleep(stagger_seconds)

    workers_done: set[int] = set()
    t_start = time.time()
    next_update = t_start + report_interval
    stop_signaled = False

    print(flush=True)

    while len(workers_done) < len(workers):
        time.sleep(2)

        for wid, p in workers:
            if wid not in workers_done and not p.is_alive():
                exitcode = p.exitcode
                if exitcode is not None:
                    err = _read_error(tmp_dir, wid)
                    if err:
                        print(f"\n  *** {err}", flush=True)
                    elif exitcode != 0:
                        print(f"  *** Worker {wid} DIED (exitcode={exitcode})", flush=True)
                    else:
                        elapsed = time.time() - t_start
                        done_ct = len(workers_done) + 1
                        if done_ct % max(1, len(workers) // 10) == 0 or len(workers) <= 8:
                            print(f"  ✓ Worker {wid} DONE ({_fmt_sec(elapsed)})", flush=True)
                workers_done.add(wid)

        # Size-based early stop for streaming runs.
        if target_bytes > 0 and not stop_signaled:
            produced = _shard_dir_bytes(shard_dir)
            if produced >= target_bytes:
                stop_path = os.path.join(tmp_dir, "STOP")
                with open(stop_path, "w") as f:
                    f.write("1\n")
                stop_signaled = True
                print(
                    f"\n  ★ target {_fmt_bytes(target_bytes)} reached "
                    f"(shards={_fmt_bytes(produced)}); signaling STOP",
                    flush=True,
                )

        now = time.time()
        if now >= next_update and len(workers_done) < len(workers):
            elapsed = now - t_start
            total_done = sum(_read_progress(tmp_dir, wid) for wid, _ in workers)
            frac = total_done / n_games if n_games > 0 else 0
            eta = (elapsed / total_done) * (n_games - total_done) if total_done > 0 else 0
            rate = total_done / (elapsed / 3600) if elapsed > 0 else 0
            shard_sz = _shard_dir_bytes(shard_dir) if shard_bytes > 0 else 0

            print(f"\n{'─' * 50}")
            print(f"  Progress: {total_done}/{n_games} games ({frac*100:.1f}%)")
            print(f"  Elapsed:  {_fmt_sec(elapsed)}")
            print(f"  ETA:      {_fmt_sec(eta)}")
            print(f"  Rate:     {rate:.0f} games/hr")
            if shard_bytes > 0:
                print(f"  Shards:   {_fmt_bytes(shard_sz)}"
                      + (f" / {_fmt_bytes(target_bytes)}" if target_bytes else ""))
            done_ct = len(workers_done)
            print(f"  Workers done:  {done_ct}/{len(workers)}")
            print(f"{'─' * 50}\n", flush=True)
            next_update = now + report_interval

    for _, p in workers:
        p.join(timeout=5)

    elapsed = time.time() - t_start
    print(f"\nAll workers done in {_fmt_sec(elapsed)}", flush=True)

    fin = unfin = 0
    for wid, _ in workers:
        spath = os.path.join(tmp_dir, f"stats_{wid}.txt")
        if os.path.exists(spath):
            try:
                a, b = open(spath).read().split()
                fin += int(a)
                unfin += int(b)
            except Exception:
                pass
    if fin + unfin:
        pct = 100.0 * fin / (fin + unfin)
        print(f"Games reaching a terminal state: {fin}/{fin + unfin} ({pct:.0f}%)", flush=True)
        if pct < 50:
            print(
                "  WARNING: most games hit the ply cap. Those samples carry no\n"
                "  terminal outcome, so the value head will train on weak teacher\n"
                "  root values. Raise --uhp-depth or --random-fraction, or pass\n"
                "  --drop-unfinished.",
                flush=True,
            )

    if shard_bytes > 0:
        print(f"Finalized shards: {_fmt_bytes(_shard_dir_bytes(shard_dir))} in {shard_dir}", flush=True)

    all_samples: list[LiteTrainingSample] = []
    if merge_final:
        print("Merging residual worker buffers...", flush=True)
        for wid, _ in workers:
            path = os.path.join(tmp_dir, f"worker_{wid}.pt")
            if os.path.exists(path):
                partial = torch.load(path, weights_only=False)
                all_samples.extend(partial)
                print(f"  Worker {wid}: {len(partial)} samples", flush=True)

        if all_samples:
            random.shuffle(all_samples)
            torch.save(all_samples, output)
            print(f"\nSaved {len(all_samples)} residual samples to {output}", flush=True)
        else:
            print("\nNo residual samples (all data already in shards).", flush=True)
    else:
        print("Skipping final merge (merge_final=False).", flush=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    _SHARED_ADAPTER = None
    _SHARD_BYTES = 0
    _SHARD_DIR = None
    return all_samples


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate MCTS self-play data in parallel")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel workers (default: auto from CPU/RAM)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="dataset.pt")
    p.add_argument(
        "--report-interval",
        type=int,
        default=300,
        help="seconds between progress updates",
    )
    p.add_argument(
        "--simulations",
        type=int,
        default=50,
        help="MCTS simulations per teacher move (default 50)",
    )
    p.add_argument(
        "--stagger-seconds",
        type=float,
        default=1.0,
        help="delay between starting workers (default 1.0; 0=off)",
    )
    p.add_argument(
        "--shard-bytes",
        type=int,
        default=0,
        help="rotate finalized shards of this many bytes (e.g. 104857600 = 100MB). 0=off",
    )
    p.add_argument(
        "--shard-dir",
        type=str,
        default=None,
        help="directory for streaming shards (default: <output_dir>/shards)",
    )
    p.add_argument(
        "--target-bytes",
        type=int,
        default=0,
        help="stop generation once total shard bytes reach this (e.g. 5368709120 ≈ 5GB)",
    )
    p.add_argument(
        "--no-merge",
        action="store_true",
        help="do not merge residual worker buffers into --output (shards only)",
    )
    p.add_argument(
        "--no-sample",
        action="store_true",
        help="do not sample moves probabilistically from visitation counts (argmax only)",
    )
    p.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["teacher", "random"],
        help=(
            "Opponent to play against (used when --random-fraction is not set). "
            "'random' produces fast, decisive teacher-vs-random games with terminal "
            "±1 outcome backfills and losing-side positions. 'teacher' is slower "
            "teacher self-play with weak MCTS root values."
        ),
    )
    p.add_argument(
        "--random-fraction",
        type=float,
        default=0.5,
        help=(
            "Blend mode: fraction of games played vs random (rest are teacher "
            "self-play). Default 0.5 so the model learns both to punish random "
            "play and to play strong positions. Set to 0.0 for pure self-play, "
            "1.0 for pure vs-random, or use --opponent for the legacy binary mode."
        ),
    )
    p.add_argument(
        "--counterfactual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="query teacher for expert label/value on ALL turns (default: True; "
             "use --no-counterfactual to label teacher turns only)",
    )
    p.add_argument(
        "--scramble-plies",
        type=int,
        default=6,
        help="play N random opening plies before teacher evaluation (default: 6)",
    )
    p.add_argument(
        "--teacher",
        type=str,
        default="uhp",
        choices=["uhp", "alphazero"],
        help=(
            "Teacher move source. 'uhp' (default) uses the native Mzinga C# "
            "engine via UHP subprocess — strong, no torch needed. 'alphazero' "
            "uses the in-repo AlphaZero model + MCTS (legacy)."
        ),
    )
    p.add_argument(
        "--uhp-depth",
        type=int,
        default=4,
        help="search depth for the UHP engine's bestmove (default: 4)",
    )
    p.add_argument(
        "--uhp-epsilon",
        type=float,
        default=0.05,
        help="epsilon-greedy exploration rate for UHP play moves (default: 0.05)",
    )
    p.add_argument(
        "--history-window",
        type=int,
        default=None,
        help=(
            "cap <history> to the last N move strings (default: None = full "
            "transcript). The full transcript grows contexts to ~2100 tokens "
            "by the midgame; attention is O(T^2), so this dominates training "
            "cost. Contexts are stored pre-tokenized, so this is baked into "
            "the dataset at generation time. Measured: 40 gives ~850-token "
            "contexts (the <features> and <legal> sections make up the rest, "
            "and <legal> scales with the branching factor)."
        ),
    )
    p.add_argument(
        "--max-plies",
        type=int,
        default=400,
        help="hard ply cap per game (default: 400)",
    )
    p.add_argument(
        "--drop-unfinished",
        action="store_true",
        help=(
            "discard samples from games that hit --max-plies. Those games get "
            "no terminal outcome backfill, so their samples carry only the "
            "teacher's weak root value and dilute the value-head signal."
        ),
    )
    args = p.parse_args()

    generate_dataset_parallel(
        n_games=args.games,
        num_workers=args.workers,
        seed=args.seed,
        output=args.output,
        report_interval=args.report_interval,
        num_simulations=args.simulations,
        stagger_seconds=args.stagger_seconds,
        shard_bytes=args.shard_bytes,
        shard_dir=args.shard_dir,
        target_bytes=args.target_bytes,
        merge_final=not args.no_merge,
        sample=not args.no_sample,
        opponent=args.opponent,
        counterfactual=args.counterfactual,
        scramble_plies=args.scramble_plies,
        teacher=args.teacher,
        uhp_depth=args.uhp_depth,
        uhp_epsilon=args.uhp_epsilon,
        random_fraction=args.random_fraction,
        history_window=args.history_window,
        max_plies=args.max_plies,
        drop_unfinished=args.drop_unfinished,
    )
