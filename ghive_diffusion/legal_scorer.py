"""Hive legal-move scorer (Phase 5).

This module is the bridge between Mzinga (legal-move authority) and
``HiveDiffusionModel`` (policy / value source). For any board position:

1. Enumerate legal moves from Mzinga (``board.get_valid_moves()``).
2. Convert each move to a canonical string (``board.try_get_move_string``).
3. Tokenize each legal move.
4. Score them with the model (``HiveDiffusionModel.score_legal_moves``).
5. Return the distribution over legal moves.

The scorer guarantees:

- Every score corresponds to a Mzinga-legal move.
- The final sampler always returns a Mzinga-legal move.
- Pass is only emitted/chosen when Mzinga says it is legal.
- Expansion-piece availability is respected (``GameType``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mzinga.core.board import Board

from .context_builder import HiveContextBuilder
from .hive_model import HiveDiffusionModel
from .tokenizer import HiveTokenizer


@dataclass
class ScoredMove:
    move_str: str
    move: object  # mzinga.core.move.Move
    score: float
    token_ids: List[int]


class HiveLegalScorer:
    """Score and rank every legal move for a given board.

    Parameters
    ----------
    model : HiveDiffusionModel
    tokenizer : HiveTokenizer
    builder : HiveContextBuilder, optional
    use_value_head : bool
        If True, also returns the value estimate from the value head.
    """

    def __init__(self, model: HiveDiffusionModel, tokenizer: HiveTokenizer,
                 builder: Optional[HiveContextBuilder] = None,
                 use_value_head: bool = False,
                 use_confidence_weight: bool = True,
                 self_cond_passes: int = 2):
        self.model = model
        self.tokenizer = tokenizer
        self.builder = builder or HiveContextBuilder(tokenizer)
        self.use_value_head = use_value_head
        self.use_confidence_weight = use_confidence_weight
        self.self_cond_passes = self_cond_passes

    # ----- per-board scoring ----------------------------------------------

    def legal_move_strings(self, board: Board) -> List[str]:
        moves = list(board.get_valid_moves())
        out: List[str] = []
        for mv in moves:
            try:
                ms = board.get_move_string(mv)
            except Exception:
                ms = None
            if ms is None:
                continue
            out.append(ms)
        return out

    def score(self, board: Board, return_probs: bool = False) -> List[ScoredMove]:
        """Return one :class:`ScoredMove` per legal move.

        ``return_probs=True`` returns the softmax-normalized probabilities
        alongside the raw scores.
        """
        device = next(self.model.parameters()).device
        legal_strs = self.legal_move_strings(board)
        if not legal_strs:
            return []

        ctx_ids = self.builder.encode(board, target_move=None)
        context_ids = torch.tensor([ctx_ids], dtype=torch.long, device=device)
        legal_ids_list = [self.tokenizer.encode_move(s) for s in legal_strs]

        with torch.no_grad():
            scores, value = self.model.score_legal_moves(
                context_ids, legal_ids_list, attn_mask=None,
                use_value_head=self.use_value_head,
                use_confidence_weight=self.use_confidence_weight,
                self_cond_passes=self.self_cond_passes,
            )
        scores = scores.squeeze(0)  # (n_moves,)
        if return_probs:
            probs = F.softmax(scores.float(), dim=-1).cpu().numpy()
        scores_np = scores.float().cpu().numpy()

        # Reconstruct the original Move objects for callers who want to play.
        legal_moves = list(board.get_valid_moves())
        move_lookup = {}
        for mv in legal_moves:
            try:
                ms = board.get_move_string(mv)
            except Exception:
                continue
            move_lookup[ms] = mv

        out: List[ScoredMove] = []
        for i, ms in enumerate(legal_strs):
            out.append(ScoredMove(
                move_str=ms,
                move=move_lookup.get(ms),
                score=float(scores_np[i]),
                token_ids=legal_ids_list[i],
            ))
        return out

    def best_move(self, board: Board, deterministic: bool = True,
                  temperature: float = 1.0) -> Tuple[str, float]:
        """Return ``(move_string, score)`` for the best legal move.

        ``deterministic=True`` picks argmax. With ``temperature=0`` or
        ``temperature=None``, also argmax. Otherwise samples from the
        softmax.
        """
        scored = self.score(board, return_probs=True)
        if not scored:
            raise RuntimeError("No legal moves")
        scores = torch.tensor([s.score for s in scored])
        if deterministic or temperature in (0, None):
            idx = int(torch.argmax(scores).item())
        else:
            probs = F.softmax(scores.float() / max(temperature, 1e-3), dim=-1)
            idx = int(torch.multinomial(probs, 1).item())
        return scored[idx].move_str, float(scores[idx].item())

    # ----- value ----------------------------------------------------------

    def value(self, board: Board) -> float:
        device = next(self.model.parameters()).device
        ctx_ids = self.builder.encode(board, target_move=None)
        context_ids = torch.tensor([ctx_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            v = self.model.encode_value(context_ids)
        return float(v.item())