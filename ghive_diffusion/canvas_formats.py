"""Candidate / PV canvas formats (Phase 8).

These are the structured text formats the model can denoise during
training. Each format is encoded into a token sequence suitable for the
decoder canvas.

Formats supported:

- :func:`format_single_move` — ``<move> wA1 /bQ <eos>``
- :func:`format_candidate_set` — ``<candidates> wA1 /bQ ; wG2 bS1- ; pass <eos>``
- :func:`format_principal_variation` — ``<pv> wA1 /bQ ; bB2 wQ- ; wG2 bQ/ ; bQ wA1- <eos>``
- :func:`format_move_with_value` — ``<move> wA1 /bQ <value> <v+2> <eos>``

The value bucket ``<v+N>`` quantizes a continuous scalar in ``[-1, 1]``
to one of ``[-4, -3, -2, -1, 0, +1, +2, +3, +4]``. Buckets outside the
range are clamped.
"""

from __future__ import annotations

from typing import List, Sequence

from .tokenizer import HiveTokenizer


VALUE_BUCKETS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


def value_bucket_token(value: float) -> str:
    """Quantize a scalar value in [-1, 1] to one of the discrete buckets."""
    if value <= -1.0:
        bucket = VALUE_BUCKETS[0]
    elif value >= 1.0:
        bucket = VALUE_BUCKETS[-1]
    else:
        # 9 buckets over [-1, 1]: each covers 2/9 ≈ 0.222
        idx = int((value + 1.0) / (2.0 / 9.0))
        idx = max(0, min(len(VALUE_BUCKETS) - 1, idx))
        bucket = VALUE_BUCKETS[idx]
    sign = "+" if bucket >= 0 else ""
    return f"<v{sign}{bucket}>"


def bucket_token_to_value(token: str) -> float:
    """Inverse of :func:`value_bucket_token`."""
    if not (token.startswith("<v") and token.endswith(">")):
        return 0.0
    try:
        return int(token[2:-1]) / 4.0
    except ValueError:
        return 0.0


def _tokens_for_move(move: str, tk: HiveTokenizer) -> List[int]:
    return tk.encode_move(move)


# ----- single move ----------------------------------------------------------

def format_single_move(move: str, tk: HiveTokenizer) -> List[int]:
    """Encode a single-move canvas: ``<move> <mv> <eos>``."""
    toks = ["<move>"] + tk._split_tokens(move) + ["<eos>"]
    return tk.encode_text(" ".join(toks), add_bos=False, add_eos=False)


# ----- candidate set --------------------------------------------------------

def format_candidate_set(moves: Sequence[str], tk: HiveTokenizer) -> List[int]:
    """Encode a candidate-set canvas."""
    parts: List[str] = ["<candidates>"]
    for i, mv in enumerate(moves):
        if i > 0:
            parts.append(";")
        parts.append(mv)
    parts.append("<eos>")
    return tk.encode_text(" ".join(parts), add_bos=False, add_eos=False)


# ----- principal variation --------------------------------------------------

def format_principal_variation(pv: Sequence[str], tk: HiveTokenizer) -> List[int]:
    """Encode a principal variation canvas: ``<pv> mv1 ; mv2 ; ... <eos>``."""
    parts: List[str] = ["<pv>"]
    for i, mv in enumerate(pv):
        if i > 0:
            parts.append(";")
        parts.append(mv)
    parts.append("<eos>")
    return tk.encode_text(" ".join(parts), add_bos=False, add_eos=False)


# ----- move with value ------------------------------------------------------

def format_move_with_value(move: str, value: float, tk: HiveTokenizer) -> List[int]:
    """Encode ``<move> <mv> <value> <vN> <eos>``."""
    parts: List[str] = ["<move>", move, "<value>", value_bucket_token(value), "<eos>"]
    return tk.encode_text(" ".join(parts), add_bos=False, add_eos=False)


# ----- dispatch ------------------------------------------------------------

CANVAS_FORMATS = ("single", "candidates", "pv", "move_value")


def format_canvas(fmt: str, *, move: str = "",
                  moves: Sequence[str] = (),
                  pv: Sequence[str] = (),
                  value: float = 0.0,
                  tk: HiveTokenizer) -> List[int]:
    """Dispatch to the right canvas formatter."""
    if fmt == "single":
        if not move:
            raise ValueError("single canvas requires move=")
        return format_single_move(move, tk)
    if fmt == "candidates":
        return format_candidate_set(moves, tk)
    if fmt == "pv":
        return format_principal_variation(pv, tk)
    if fmt == "move_value":
        return format_move_with_value(move, value, tk)
    raise ValueError(f"Unknown canvas format {fmt!r}; choices: {CANVAS_FORMATS}")