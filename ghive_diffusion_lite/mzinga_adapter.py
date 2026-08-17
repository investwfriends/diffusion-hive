"""Adapter: Mzinga's pretrained AlphaZero + MCTS as a move-policy callable.

``MzingaMCTSAdapter`` loads the AlphaZero-trained ``HivePolicyValue``
from ``Mzinga/colab/mzinga_alphazero_final.pt``, wraps it in Mzinga's
``MCTS``, and exposes a ``board -> move_string`` interface compatible
with ``SelfPlayGenerator.move_policy``.

Each call runs MCTS (50 simulations by default).  After a search the
adapter exposes:

- ``last_label_move`` — argmax-by-visits move (clean supervised target)
- ``last_play_move`` — sampled or argmax move used to advance the game
- ``last_value`` — visit-weighted root value (position estimate for
  side-to-move), **not** Q of the sampled action

MCTS restores the board to its original state via undo, so it is safe to
call repeatedly during game generation.  We still search on a clone so
the caller's board is never mutated.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from mzinga.core.board import Board
from mzinga.rl.mcts import MCTS
from mzinga.rl.model import HivePolicyValue

_ALPHAZERO_PATH = None


def _find_checkpoint():
    global _ALPHAZERO_PATH
    if _ALPHAZERO_PATH is not None:
        return _ALPHAZERO_PATH
    start = os.path.dirname(os.path.abspath(__file__))
    root = start
    rel = os.path.join("Mzinga", "colab", "mzinga_alphazero_final.pt")
    for _ in range(5):
        root = os.path.dirname(root)
        candidate = os.path.join(root, rel)
        if os.path.isfile(candidate):
            _ALPHAZERO_PATH = candidate
            return candidate
        if root == "/" or root == "":
            break
    raise FileNotFoundError(f"Could not find {rel} from {start}")


def _load_model_state_dict(checkpoint_path: str) -> dict:
    """Load only weights — drop optimizer/scheduler to keep RSS small."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        # Drop heavy training baggage immediately (optimizer is ~1.1 MB pickled).
        del ckpt
        return state
    return ckpt


def _root_value_from_tree(mcts: MCTS) -> float:
    """Visit-weighted mean of root child W/N totals = position value.

    Equivalent to ``sum(W_a) / sum(N_a)`` over root children with N>0.
    This is the side-to-move estimate, not Q of a single sampled action.
    """
    root_key = getattr(mcts, "_root_key", None)
    tree = getattr(mcts, "tree", None)
    if root_key is None or not tree or root_key not in tree:
        return 0.0
    total_n = 0.0
    total_w = 0.0
    for n, w, _p_cur, _p_root in tree[root_key].values():
        if n > 0:
            total_n += float(n)
            total_w += float(w)
    if total_n <= 0:
        return 0.0
    return float(total_w / total_n)


class MzingaMCTSAdapter:
    """Wraps Mzinga's pretrained AlphaZero + MCTS as a board→move_string callable.

    Parameters
    ----------
    device : str
        ``"cpu"`` or ``"mps"``.  MPS recommended for Apple Silicon.
    num_simulations : int
        MCTS rollouts per move (default 50).  More = stronger but slower.
        Typical games (~30 plies) take 2–10s on CPU depending on this value.
    sample : bool
        If True, ``last_play_move`` / ``__call__`` sample from visit counts.
        ``last_label_move`` is always the argmax-by-visits move.
    """

    def __init__(self, device: str = "cpu", num_simulations: int = 50, sample: bool = True):
        self.model = HivePolicyValue(hidden_dim=64, num_blocks=1)
        checkpoint_path = _find_checkpoint()
        state = _load_model_state_dict(checkpoint_path)
        self.model.load_state_dict(state)
        self.model = self.model.to(device)
        self.model.eval()
        # Keep weights read-only so fork COW pages stay shared across workers.
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mcts = MCTS(self.model, num_simulations=num_simulations)
        self._device = device
        self.num_simulations = num_simulations
        self.sample = sample

        # Populated by the most recent evaluate()/__call__().
        self.last_value: float = 0.0
        self.last_label_move: Optional[str] = None
        self.last_play_move: Optional[str] = None

    def evaluate(self, board: Board) -> Tuple[Optional[str], Optional[str], float]:
        """Run one MCTS search; return ``(label_move, play_move, root_value)``.

        - ``label_move``: highest-visit root action (supervised target).
        - ``play_move``: sampled from π if ``self.sample`` else same as label.
        - ``root_value``: visit-weighted root value for side-to-move.
        """
        pi, _pi_probs, best_a = self.mcts.search(board.clone())
        valid = list(board.get_valid_moves())
        if not valid:
            self.last_value = 0.0
            self.last_label_move = "pass"
            self.last_play_move = "pass"
            return self.last_label_move, self.last_play_move, self.last_value

        n_valid = len(valid)
        label_a = int(best_a) if best_a is not None and int(best_a) < n_valid else 0
        # Prefer explicit visit-argmax if best_a is out of range or missing visits.
        root_key = getattr(self.mcts, "_root_key", None)
        if root_key in self.mcts.tree:
            visited = {
                a: n
                for a, (n, _w, _pc, _pr) in self.mcts.tree[root_key].items()
                if n > 0 and a < n_valid
            }
            if visited:
                label_a = max(visited, key=visited.get)

        try:
            label_str = board.get_move_string(valid[label_a])
        except Exception:
            label_str = None

        if self.sample:
            probs = np.asarray(pi[:n_valid], dtype=np.float64)
            sum_probs = float(probs.sum())
            if sum_probs > 0:
                probs = probs / sum_probs
                play_a = int(np.random.choice(n_valid, p=probs))
            else:
                play_a = label_a
            try:
                play_str = board.get_move_string(valid[play_a])
            except Exception:
                play_str = label_str
        else:
            play_str = label_str

        self.last_value = _root_value_from_tree(self.mcts)
        self.last_label_move = label_str
        self.last_play_move = play_str
        return self.last_label_move, self.last_play_move, self.last_value

    def __call__(self, board: Board) -> str:
        """Return the play move (sampled or argmax). Prefer ``evaluate`` for labels."""
        _label, play, _value = self.evaluate(board)
        if play is None:
            return "pass"
        return play
