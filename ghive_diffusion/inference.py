"""Inference: fast play and MCTS-guided search (Phase 10).

Two modes are supported:

- :class:`FastPlayer` — score every legal move and pick argmax / sample.
  Now includes value-head lookahead (1-ply minimax), pondering with aux heads,
  and diffusion-based generative candidate validation.

- :class:`MCTSPlayer` — MCTS that uses the diffusion model as both policy
  prior and value evaluator.  Includes phase-guided progressive temperature.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mzinga.core.board import Board
from mzinga.core.move import Move

from .context_builder import HiveContextBuilder
from .hive_model import HiveDiffusionModel
from .legal_scorer import HiveLegalScorer, ScoredMove
from .tokenizer import HiveTokenizer


# ---------------------------------------------------------------------------
# Helper: determine game phase from turn number
# ---------------------------------------------------------------------------

def _game_phase(turn: int) -> str:
    if turn <= 10:
        return "open"
    elif turn <= 30:
        return "midgame"
    return "endgame"


# ---------------------------------------------------------------------------
# FastPlayer with value lookahead, pondering, and generative validation
# ---------------------------------------------------------------------------

class FastPlayer:
    """Score legal moves and pick the best.

    Improvements over the baseline scorer:

    - **value-head lookahead** (``lookahead_k`` > 0): for the top-k scored
      moves, simulate playing each and evaluate the resulting position via
      the value head.  Adjusts scores::

          adjusted = policy_score + 0.2 * (value_after_move - value_now)

    - **pondering**: call :meth:`ponder` after the opponent plays to pre-
      compute context encoding + aux head predictions.  Subsequent
      :meth:`play` calls skip the encoder pass and use aux signals
      (e.g. queen threatened → boost queen-saving moves).

    - **diffusion candidate validation** (``diffusion_candidates``): run a
      short generative denoising pass and boost any legal move that matches
      the model's own generative output.

    Parameters
    ----------
    model : HiveDiffusionModel
    tokenizer : HiveTokenizer
    builder : HiveContextBuilder, optional
    deterministic : bool
    temperature : float
    lookahead_k : int
        Number of top-scored moves to evaluate with 1-ply value lookahead.
        0 disables.
    lookahead_weight : float
        Weight of the value difference in the adjusted score.
    diffusion_candidates : bool
        If True, run a short generative denoising pass.
    """

    def __init__(self, model: HiveDiffusionModel, tokenizer: HiveTokenizer,
                 builder: Optional[HiveContextBuilder] = None,
                 deterministic: bool = True,
                 temperature: float = 1.0,
                 lookahead_k: int = 3,
                 lookahead_weight: float = 0.2,
                 diffusion_candidates: bool = True):
        self.scorer = HiveLegalScorer(
            model, tokenizer, builder,
            use_value_head=True,
            use_confidence_weight=True,
            self_cond_passes=2,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.builder = builder or HiveContextBuilder(tokenizer)
        self.deterministic = deterministic
        self.temperature = temperature
        self.lookahead_k = lookahead_k
        self.lookahead_weight = lookahead_weight
        self.diffusion_candidates = diffusion_candidates

        # Pondering cache — keyed by board zobrist_key
        self._ponder: Dict[int, dict] = {}

    # ── scoring ──────────────────────────────────────────────────────────

    def score(self, board: Board) -> List[ScoredMove]:
        return self.scorer.score(board, return_probs=True)

    def value(self, board: Board) -> float:
        return self.scorer.value(board)

    # ── pondering (Tier 1 #2) ────────────────────────────────────────────

    def ponder(self, board: Board) -> None:
        """Pre-compute context encoding + aux head predictions.

        Call this after the opponent moves.  The cached results are used
        by :meth:`play` to skip redundant encoder passes and to bias move
        selection with aux head signals.
        """
        if board.game_is_over:
            return
        device = next(self.model.parameters()).device
        ctx_ids = self.builder.encode(board, target_move=None)
        context_ids = torch.tensor([ctx_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            v = self.model.encode_value(context_ids)
            aux = self.model.forward_aux_heads(context_ids)
            # Compute aux head argmax predictions (int labels)
            aux_labels = [int(a.argmax(dim=-1).item()) for a in aux]

        self._ponder[board.zobrist_key] = {
            "value": float(v.item()),
            "aux_labels": aux_labels,       # list of 9 ints
            "context_ids": ctx_ids,
        }

    def _ponder_for(self, board: Board) -> Optional[dict]:
        return self._ponder.get(board.zobrist_key)

    # ── generative candidate validation (Tier 2 #6) ──────────────────────

    def _diffusion_candidate_boost(self, board: Board,
                                    scored: List[ScoredMove]) -> List[ScoredMove]:
        """Run a short generative denoising pass and boost matching legal moves."""
        try:
            device = next(self.model.parameters()).device
            ctx_ids = self.builder.encode(board, target_move=None)
            input_ids = torch.tensor([ctx_ids], dtype=torch.long, device=device)

            # Shorter denoising — we just need a hint, not a perfect generation.
            result = self.model.generate(
                input_ids, max_new_tokens=self.model.cfg.canvas_length,
                max_denoising_steps=16,
                t_min=0.3, t_max=0.7,
                entropy_bound=0.15,
                confidence_threshold=0.01,
                stability_threshold=1,
            )
            # Decode the generated canvas tokens that follow the context.
            gen_ids = result.sequences[0, input_ids.size(1):].tolist()
            gen_str = self.tokenizer.decode(gen_ids, skip_special=True).strip()
            if not gen_str:
                return scored

            # Check against legal move strings.
            for s in scored:
                if s.move_str == gen_str or gen_str.startswith(s.move_str):
                    s.score += 1.0  # boost
                    break
        except Exception:
            pass
        return scored

    # ── play (Tier 1 #1 + #2 + #6) ───────────────────────────────────────

    def play(self, board: Board, deterministic: Optional[bool] = None,
             temperature: Optional[float] = None) -> Move:
        deterministic = self.deterministic if deterministic is None else deterministic
        temperature = self.temperature if temperature is None else temperature

        scored = self.scorer.score(board, return_probs=True)
        if not scored:
            raise RuntimeError("No legal moves")

        # ── aux head biasing from ponder cache ──
        ponder = self._ponder_for(board)
        if ponder is not None:
            aux_labels = ponder["aux_labels"]
            # Head 6: queen_surround_count (0=0, 1=1, 2=2, 3=3+)
            # Head 3: queen_placement_required (0=no, 1=yes)
            queen_threatened = aux_labels[6] >= 2
            queen_must_place = aux_labels[3] == 1

            for s in scored:
                ms = s.move_str
                # Boost queen-placement moves when queen must be placed
                if queen_must_place:
                    if any(p in ms for p in ("wQ", "bQ", "wM", "bM", "wL", "bL", "wP", "bP")):
                        s.score += 1.0
                # Deprioritize non-queen moves when queen is threatened
                if queen_threatened:
                    is_queen_move = any(p in ms for p in ("wQ", "bQ"))
                    if not is_queen_move:
                        s.score -= 0.5

        # ── diffusion candidate validation ──
        if self.diffusion_candidates:
            scored = self._diffusion_candidate_boost(board, scored)

        # ── value-head 1-ply lookahead for top-k ──
        if self.lookahead_k > 0 and len(scored) > 1:
            sorted_moves = sorted(scored, key=lambda s: s.score, reverse=True)
            lookahead_n = min(self.lookahead_k, len(sorted_moves))
            current_value = self.value(board)

            for i in range(lookahead_n):
                s = sorted_moves[i]
                if s.move is None:
                    continue
                try:
                    sim_board = board.clone()
                    ms = sim_board.get_move_string(s.move)
                    sim_board.trusted_play(s.move, ms)
                    next_value = self.value(sim_board)
                    s.score += self.lookahead_weight * ((-next_value) - current_value)
                except Exception:
                    continue

        # ── select move ──
        scores = torch.tensor([s.score for s in scored])
        if deterministic or temperature in (0, None):
            idx = int(torch.argmax(scores).item())
        else:
            probs = F.softmax(scores.float() / max(temperature, 1e-3), dim=-1)
            idx = int(torch.multinomial(probs, 1).item())

        move_str = scored[idx].move_str
        for mv in board.get_valid_moves():
            try:
                if board.get_move_string(mv) == move_str:
                    return mv
            except Exception:
                continue
        # Fallback
        legal = list(board.get_valid_moves())
        if not legal:
            raise RuntimeError("No legal moves")
        return legal[0]


# ---------------------------------------------------------------------------
# ValuePlayer: pure 1-ply value-greedy player
# ---------------------------------------------------------------------------

class ValuePlayer:
    """Play the move that maximizes the value-head score of the successor.

    This player ignores the policy head.  For each legal move it clones the
    board, plays the move, and evaluates the resulting position with the value
    head.  Because the value head is trained from the side-to-move perspective,
    the successor value is from the opponent's point of view; we therefore play
    ``argmax(-value(successor))``, i.e. the move that leaves the opponent in the
    worst position.
    """

    name = "value"

    def __init__(self, model: HiveDiffusionModel, tokenizer: HiveTokenizer,
                 builder: Optional[HiveContextBuilder] = None,
                 deterministic: bool = True,
                 temperature: float = 1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.builder = builder or HiveContextBuilder(tokenizer)
        self.deterministic = deterministic
        self.temperature = temperature

    def value(self, board: Board) -> float:
        """Side-to-move value estimate for *board*."""
        if board.game_is_over:
            state = board.board_state
            from mzinga.core.enums import BoardState
            if state == BoardState.Draw:
                return 0.0
            # Game over and not a draw: the side to move has already lost.
            return -1.0
        device = next(self.model.parameters()).device
        ctx_ids = self.builder.encode(board, target_move=None)
        context_ids = torch.tensor([ctx_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            v = self.model.encode_value(context_ids)
        return float(v.item())

    def play(self, board: Board) -> Move:
        """Return the legal move with the best (highest) successor score."""
        legal_moves = list(board.get_valid_moves())
        if not legal_moves:
            from mzinga.core.move import PASS_MOVE
            return PASS_MOVE

        best_move = None
        best_score = float("-inf")
        candidates: List[Tuple[Move, float]] = []

        for mv in legal_moves:
            try:
                ms = board.get_move_string(mv)
                if not ms:
                    continue
                sim_board = board.clone()
                sim_board.trusted_play(mv, ms)
                # Successor value is from the opponent's perspective.  A good
                # move leaves the opponent with a low (ideally -1) value, so we
                # negate it when scoring from the current player's view.
                score = -self.value(sim_board)
                candidates.append((mv, score))
                if score > best_score:
                    best_score = score
                    best_move = mv
            except Exception:
                continue

        if not candidates:
            return legal_moves[0]

        if self.deterministic or self.temperature in (0, None):
            return best_move if best_move is not None else candidates[0][0]

        # Temperature sampling over the raw value scores.
        moves, scores = zip(*candidates)
        probs = F.softmax(torch.tensor(scores) / max(self.temperature, 1e-3), dim=-1)
        idx = int(torch.multinomial(probs, 1).item())
        return moves[idx]


# ---------------------------------------------------------------------------
# MCTSPlayer with phase-guided progressive temperature
# ---------------------------------------------------------------------------

class MCTSPlayer:
    """Run a lightweight MCTS that uses the diffusion model for priors/value.

    The tree is implemented in numpy for portability. Each leaf at non-terminal
    nodes queries the model for legal-move priors (via :class:`HiveLegalScorer`)
    and a scalar value estimate. The first move returned is the highest-visit
    root child.

    Tier 1 #3 — phase-guided progressive temperature: the opening phase uses
    higher temperature for exploration, the endgame switches to deterministic
    selection.
    """

    def __init__(self, model: HiveDiffusionModel, tokenizer: HiveTokenizer,
                 builder: Optional[HiveContextBuilder] = None,
                 num_simulations: int = 50,
                 c_puct: float = 1.4,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 progressive_temperature: bool = True):
        self.fast = FastPlayer(model, tokenizer, builder, lookahead_k=0,
                               diffusion_candidates=False)
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.progressive_temperature = progressive_temperature
        self._tree: dict[int, dict[int, Tuple[float, float, float]]] = {}
        self._sim_path: List[Tuple[int, int]] = []

    def _move_index(self, board: Board, move: Move) -> int:
        moves = list(board.get_valid_moves())
        for i, mv in enumerate(moves):
            if mv == move:
                return i
        return -1

    def _select_action(self, board_key: int) -> Optional[int]:
        children = self._tree.get(board_key, {})
        if not children:
            return None
        total_N = sum(c[0] for c in children.values())
        best_a, best_ucb = None, -float("inf")
        for a, (N, W, P) in children.items():
            Q = W / N if N > 0 else 0.0
            ucb = Q + self.c_puct * P * np.sqrt(total_N + 1e-6) / (1.0 + N)
            if ucb > best_ucb:
                best_ucb = ucb
                best_a = a
        return best_a

    def _expand(self, board: Board) -> float:
        """Expand a leaf. Returns the leaf's value estimate."""
        scored = self.fast.score(board)
        priors = np.array([np.exp(s.score) for s in scored], dtype=np.float64)
        if priors.sum() > 0:
            priors = priors / priors.sum()
        else:
            priors = np.ones(len(scored)) / max(1, len(scored))

        key = board.zobrist_key
        self._tree[key] = {}
        for i, s in enumerate(scored):
            self._tree[key][i] = (0.0, 0.0, float(priors[i]))
        value = self.fast.value(board)
        return value

    def _phase_temperature(self, turn: int) -> Tuple[bool, float]:
        """Return (deterministic, temperature) for the given turn number.

        Tier 1 #3 — opening is exploratory, endgame is exploitative.
        """
        phase = _game_phase(turn)
        if phase == "open":
            return False, 1.2
        elif phase == "midgame":
            return False, 0.8
        else:  # endgame
            return True, 0.0

    def search(self, board: Board) -> Move:
        """Run ``num_simulations`` and return the best move."""
        self._tree = {}
        self._sim_path = []

        scored = self.fast.score(board)
        if not scored:
            raise RuntimeError("No legal moves at root")

        priors = np.array([np.exp(s.score) for s in scored], dtype=np.float64)
        priors = priors / priors.sum() if priors.sum() > 0 else np.ones_like(priors) / len(priors)
        if len(scored) > 1:
            noise = np.random.default_rng().dirichlet([self.dirichlet_alpha] * len(scored))
            priors = (1 - self.dirichlet_epsilon) * priors + self.dirichlet_epsilon * noise

        root_key = board.zobrist_key
        self._tree[root_key] = {i: (0.0, 0.0, float(priors[i])) for i in range(len(scored))}

        for _ in range(self.num_simulations):
            self._sim_path.clear()
            sim_board = board.clone()
            sim_key = root_key

            while sim_key in self._tree and self._tree[sim_key] and not sim_board.game_is_over:
                best_a = self._select_action(sim_key)
                if best_a is None:
                    break
                legal = list(sim_board.get_valid_moves())
                if best_a >= len(legal):
                    break
                mv = legal[best_a]
                try:
                    ms = sim_board.get_move_string(mv)
                except ValueError:
                    # Mzinga can fail to stringify some valid moves; stop the
                    # descent here and evaluate this position as the leaf.
                    self._sim_path.append((sim_key, best_a))
                    break
                self._sim_path.append((sim_key, best_a))
                sim_board.trusted_play(mv, ms)
                sim_key = sim_board.zobrist_key

            if sim_board.game_is_over:
                from mzinga.core.enums import BoardState, PlayerColor
                state = sim_board.board_state
                # Terminal values must be from the side-to-move's perspective,
                # matching the value head's training convention (see
                # ``game_outcome_value`` in ghive_diffusion/dataset.py).
                if state == BoardState.Draw:
                    leaf_value = 0.0
                elif state == BoardState.WhiteWins:
                    leaf_value = 1.0 if sim_board.current_color == PlayerColor.White else -1.0
                elif state == BoardState.BlackWins:
                    leaf_value = 1.0 if sim_board.current_color == PlayerColor.Black else -1.0
                else:
                    leaf_value = 0.0
            else:
                if sim_key not in self._tree or not self._tree[sim_key]:
                    leaf_value = self._expand(sim_board)
                else:
                    leaf_value = self.fast.value(sim_board)

            # leaf_value is from the leaf's side-to-move perspective.  Each
            # edge stores value from the perspective of the player to move at
            # that node, so negate once before the walk and flip each level.
            v = -leaf_value
            for prev_key, action in reversed(self._sim_path):
                N, W, P = self._tree[prev_key][action]
                N += 1.0
                W += v
                self._tree[prev_key][action] = (N, W, P)
                v = -v

        # ── phase-guided temperature for root move selection ──
        children = self._tree[root_key]
        if not children:
            return list(board.get_valid_moves())[0]

        deterministic, temp = self._phase_temperature(board.current_turn) if self.progressive_temperature else (False, 1.0)

        if deterministic:
            best_a = max(children.keys(), key=lambda a: children[a][0])
        else:
            visits = np.array([children[a][0] for a in children.keys()], dtype=np.float64)
            probs = visits ** (1.0 / max(temp, 1e-3))
            probs = probs / probs.sum()
            best_a = list(children.keys())[int(np.random.default_rng().choice(len(probs), p=probs))]

        legal = list(board.get_valid_moves())
        return legal[best_a]


# ---------------------------------------------------------------------------
# Option 4: plan-conditioned diffusion player
# ---------------------------------------------------------------------------

class PlanPlayer:
    """Generate a multi-move plan via the diffusion model and play its first,
    Mzinga-validated move.

    This finally gives the diffusion/LLM half of the model a real job at
    inference time: instead of scoring one move at a time (which cannot plan
    ahead), it denoises the next several plies as a sequence and commits to the
    first move of that plan. If the generated first move is not a legal move,
    it falls back to the value-lookahead policy scorer.
    """

    name = "plan"

    def __init__(self, model, tokenizer, builder=None,
                 max_new_tokens=32, max_denoising_steps=24):
        self.model = model
        self.tokenizer = tokenizer
        self.builder = builder or HiveContextBuilder(tokenizer)
        self.max_new_tokens = max_new_tokens
        self.max_denoising_steps = max_denoising_steps
        self.fast = FastPlayer(model, tokenizer, builder,
                               lookahead_k=3, lookahead_weight=0.5,
                               diffusion_candidates=False)

    def play(self, board: Board) -> Move:
        device = next(self.model.parameters()).device
        ctx_ids = self.builder.encode(board, target_move=None)
        input_ids = torch.tensor([ctx_ids], dtype=torch.long, device=device)

        gen = []
        try:
            out = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                max_denoising_steps=self.max_denoising_steps,
            )
            gen = out.sequences[0, input_ids.size(1):].tolist()
        except Exception:
            gen = []

        eos = getattr(self.model.cfg, "eos_token_id", 1)
        first = []
        for t in gen:
            if t == eos:
                break
            first.append(t)

        move_str = self.tokenizer.decode(first, skip_special=True).strip() if first else ""
        if move_str:
            for mv in board.get_valid_moves():
                try:
                    s = board.get_move_string(mv)
                    if s and s.strip() == move_str:
                        return mv
                except Exception:
                    continue

        # Fallback: value-lookahead policy scorer.
        return self.fast.play(board)
