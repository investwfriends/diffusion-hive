"""Hive tokenizer (Phase 4).

Covers every token class required by the adaptation plan:

- special: ``<pad>``, ``<bos>``, ``<eos>``, ``<unk>``, ``<mask>``, ``<sep>``
- task tags: ``<state>``, ``<history>``, ``<legal>``, ``<move>``, ``<pv>``, ``<value>``, ``<candidates>``, ``<features>``
- game types: ``Base``, ``Base+M``, ..., ``Base+MLP``
- board states: ``NotStarted``, ``InProgress``, ``Draw``, ``WhiteWins``, ``BlackWins``
- colors: ``White``, ``Black``
- pieces: every ``PieceName`` (``wQ``, ``wS1``, ..., ``bP``)
- separators/operators: ``;``, space, ``[``, ``]``, ``/``, ``\\``, ``-``
- move literals: ``pass``
- small numeric vocab for counts/turns (0..63)
- discrete value buckets ``<v-4>``..``<v+4>`` (see Phase 8)

The tokenizer exposes a small API:

- :meth:`HiveTokenizer.encode_move` for individual moves
- :meth:`HiveTokenizer.encode_text` for arbitrary canonical text
- :meth:`HiveTokenizer.decode` for ids -> string
- :meth:`HiveTokenizer.encode_context` for a structured ``HiveContext``

Unknown input is mapped to ``<unk>`` (never ``<pad>``) and the rate is
tracked via ``unknown_count`` / ``total_count``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .hive_config import HiveDiffusionConfig


# ---------------------------------------------------------------------------
# Token classes
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>", "<sep>"]
TASK_TOKENS = [
    "<state>", "<features>", "<history>", "<legal>",
    "<move>", "<pv>", "<value>", "<candidates>",
]
GAME_TYPE_TOKENS = [
    "Base", "Base+M", "Base+L", "Base+P",
    "Base+ML", "Base+MP", "Base+LP", "Base+MLP",
]
BOARD_STATE_TOKENS = ["NotStarted", "InProgress", "Draw", "WhiteWins", "BlackWins"]
COLOR_TOKENS = ["White", "Black"]
SEPARATOR_TOKENS = [";", "[", "]", "/", "\\", "-", " "]
MOVE_LITERALS = ["pass"]

# Numeric vocabularies
NUMERIC_TOKENS = [f"n{i}" for i in range(64)]                # counts/turns
VALUE_BUCKET_TOKENS = [f"<v{i:+d}>" for i in range(-4, 5)]    # -4..+4
FEATURE_TOKENS = [                                             # <features> section keys/values
    "white_queen_in_play",
    "black_queen_in_play",
    "last",
    "turn",
    "yes",
    "no",
]
PIECE_TOKENS = [
    "wQ", "wS1", "wS2", "wB1", "wB2",
    "wG1", "wG2", "wG3",
    "wA1", "wA2", "wA3",
    "wM", "wL", "wP",
    "bQ", "bS1", "bS2", "bB1", "bB2",
    "bG1", "bG2", "bG3",
    "bA1", "bA2", "bA3",
    "bM", "bL", "bP",
]


@dataclass
class HiveTokenizer:
    """Hive-specific tokenizer.

    The tokenizer is deterministic: same input always yields same tokens.
    Unknown tokens are mapped to ``<unk>`` and counted (never to ``<pad>``).
    """

    cfg: HiveDiffusionConfig = field(default_factory=HiveDiffusionConfig)
    _vocab: List[str] = field(default_factory=list, init=False)
    _token_to_id: Dict[str, int] = field(default_factory=dict, init=False)
    _id_to_token: List[str] = field(default_factory=list, init=False)
    unknown_count: int = field(default=0, init=False)
    total_count: int = field(default=0, init=False)

    # Stable token-id reservations (must match SPECIAL_TOKENS order)
    PAD_ID: int = 0
    UNK_ID: int = 1
    BOS_ID: int = 2
    EOS_ID: int = 3
    MASK_ID: int = 4
    SEP_ID: int = 5

    def __post_init__(self):
        self._build_vocab()

    # ----- vocab construction ---------------------------------------------

    def _build_vocab(self) -> None:
        vocab: List[str] = []
        # 0..5: specials in fixed positions
        for tok in SPECIAL_TOKENS:
            vocab.append(tok)
        # 6..: task tags
        for tok in TASK_TOKENS:
            vocab.append(tok)
        for tok in GAME_TYPE_TOKENS:
            vocab.append(tok)
        for tok in BOARD_STATE_TOKENS:
            vocab.append(tok)
        for tok in COLOR_TOKENS:
            vocab.append(tok)
        for tok in PIECE_TOKENS:
            vocab.append(tok)
        for tok in SEPARATOR_TOKENS:
            vocab.append(tok)
        for tok in MOVE_LITERALS:
            vocab.append(tok)
        for tok in NUMERIC_TOKENS:
            vocab.append(tok)
        for tok in VALUE_BUCKET_TOKENS:
            vocab.append(tok)
        for tok in FEATURE_TOKENS:
            vocab.append(tok)

        # Resize model embedding to fit the vocab exactly.
        self.cfg.vocab_size = len(vocab)
        self._vocab = vocab
        self._token_to_id = {t: i for i, t in enumerate(vocab)}
        self._id_to_token = list(vocab)

    # ----- properties -----------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def pad_id(self) -> int:
        return self.PAD_ID

    @property
    def bos_id(self) -> int:
        return self.BOS_ID

    @property
    def eos_id(self) -> int:
        return self.EOS_ID

    @property
    def unk_id(self) -> int:
        return self.UNK_ID

    @property
    def mask_id(self) -> int:
        return self.MASK_ID

    @property
    def sep_id(self) -> int:
        return self.SEP_ID

    @property
    def unknown_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.unknown_count / self.total_count

    # ----- token <-> id ---------------------------------------------------

    def token_to_id(self, tok: str) -> int:
        self.total_count += 1
        if tok not in self._token_to_id:
            self.unknown_count += 1
            return self.UNK_ID
        return self._token_to_id[tok]

    def id_to_token(self, idx: int) -> str:
        if 0 <= idx < len(self._id_to_token):
            return self._id_to_token[idx]
        return self.UNK_ID

    def is_known(self, tok: str) -> bool:
        return tok in self._token_to_id

    def known_pieces(self) -> List[str]:
        return list(PIECE_TOKENS)

    def known_game_types(self) -> List[str]:
        return list(GAME_TYPE_TOKENS)

    # ----- per-game-type piece availability (NEXT_STEPS 1.4) ----------------

    # Maps piece token string -> PieceName, used to check game-type eligibility.
    _PIECE_NAME_MAP = None  # lazily built on first call

    @classmethod
    def _build_piece_name_map(cls):
        from mzinga.core.enums import PieceName
        mapping = {}
        for pn in PieceName:
            if pn in (PieceName.INVALID, PieceName.NumPieceNames):
                continue
            mapping[pn.name] = pn
        cls._PIECE_NAME_MAP = mapping

    def piece_ids_for_game_type(self, game_type_str: str) -> set:
        """Return the set of piece token IDs valid for *game_type_str*.

        *game_type_str* is one of ``GAME_TYPE_TOKENS`` (e.g. ``"Base"``,
        ``"Base+MLP"``).
        """
        from mzinga.core.enums import GameType, PieceName, piece_name_is_enabled_for_game_type
        if self._PIECE_NAME_MAP is None:
            self._build_piece_name_map()

        # Reverse-lookup the GameType from its token string.
        gt_map = {v: k for k, v in zip(
            [GameType.Base, GameType.BaseM, GameType.BaseL, GameType.BaseP,
             GameType.BaseML, GameType.BaseMP, GameType.BaseLP, GameType.BaseMLP],
            GAME_TYPE_TOKENS)}
        gt = gt_map.get(game_type_str, GameType.Base)

        valid_ids = set()
        for piece_str, piece_id in self._token_to_id.items():
            if piece_str not in PIECE_TOKENS:
                continue
            pn = self._PIECE_NAME_MAP.get(piece_str)
            if pn is None:
                continue
            if piece_name_is_enabled_for_game_type(pn, gt):
                valid_ids.add(piece_id)
        return valid_ids

    def piece_availability_mask(self, game_type_str: str) -> torch.Tensor:
        """Return a boolean mask over the full vocab.

        ``True`` = token is a piece that is legal for *game_type_str*.
        Non-piece tokens are ``False``.
        """
        import torch as _torch
        mask = _torch.zeros(self.vocab_size, dtype=_torch.bool)
        for pid in self.piece_ids_for_game_type(game_type_str):
            mask[pid] = True
        return mask

    def illegal_piece_mask(self, game_type_str: str) -> torch.Tensor:
        """Return a boolean mask of piece tokens NOT valid for *game_type_str*.

        Useful for masking diffusion softmax logits.
        """
        import torch as _torch
        mask = _torch.zeros(self.vocab_size, dtype=_torch.bool)
        all_piece_ids = set()
        for p in PIECE_TOKENS:
            if p in self._token_to_id:
                all_piece_ids.add(self._token_to_id[p])
        valid_ids = self.piece_ids_for_game_type(game_type_str)
        for pid in all_piece_ids - valid_ids:
            mask[pid] = True
        return mask

    def illegal_piece_ids_for_game_type(self, game_type_str: str) -> set:
        """Return the set of piece token IDs NOT valid for *game_type_str*."""
        all_piece_ids = set()
        for p in PIECE_TOKENS:
            if p in self._token_to_id:
                all_piece_ids.add(self._token_to_id[p])
        valid_ids = self.piece_ids_for_game_type(game_type_str)
        return all_piece_ids - valid_ids

    # ----- high-level encoders --------------------------------------------

    def _split_tokens(self, text: str) -> List[str]:
        """Whitespace- and operator-aware tokenization.

        Operators (``/``, ``\\``, ``-``, ``;``, ``[``, ``]``) become their
        own tokens. Spaces between non-space tokens are preserved as
        ``" "`` atomic tokens so the original spacing round-trips.
        """
        out: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch.isspace():
                # Emit a single space token (we'll filter empties later)
                if out and out[-1] != " ":
                    out.append(" ")
                i += 1
                continue
            if ch in ("/", "\\", "-", ";", "[", "]"):
                out.append(ch)
                i += 1
                continue
            # Accumulate a non-operator token (piece name, color, etc.)
            j = i
            while j < n and not text[j].isspace() and text[j] not in ("/", "\\", "-", ";", "[", "]"):
                j += 1
            out.append(text[i:j])
            i = j
        # Collapse leading/trailing/duplicate spaces but keep internal ones
        cleaned: List[str] = []
        for t in out:
            if t == " " and (not cleaned or cleaned[-1] == " "):
                continue
            cleaned.append(t)
        if cleaned and cleaned[-1] == " ":
            cleaned.pop()
        if cleaned and cleaned[0] == " ":
            cleaned.pop(0)
        return cleaned

    def encode_text(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        toks = self._split_tokens(text)
        ids: List[int] = []
        if add_bos:
            ids.append(self.BOS_ID)
        for t in toks:
            if t == " ":
                ids.append(self.SEP_ID)
            else:
                ids.append(self.token_to_id(t))
        if add_eos:
            ids.append(self.EOS_ID)
        return ids

    def encode_move(self, move_str: str) -> List[int]:
        """Tokenize a single Hive move string (no BOS/EOS).

        Examples: ``wB1``, ``bQ/``, ``wA1 /bQ``, ``pass``.
        """
        return self.encode_text(move_str, add_bos=False, add_eos=True)

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        """Decode an id sequence back into a Hive-style string.

        Glue rules for operators (``/``, ``\\``, ``-``):

        - If the operator is preceded by a space (``" "``), it is a
          *leading* operator that should glue to the next piece
          (e.g. ``wA1 /bQ``).
        - Otherwise it is a *trailing* operator that should glue to the
          previous piece (e.g. ``bQ/``).

        A small state machine tracks whether the last emitted item is
        an operator waiting to be glued to the next piece.
        """
        specials = set(SPECIAL_TOKENS + TASK_TOKENS)
        glue_ops = {"/", "\\", "-"}
        out_parts: List[str] = []
        pending_leading_op: Optional[str] = None  # operator waiting for next piece

        def _flush_pending_op_to_glue():
            """If we have a leading op waiting, no-op; it's already in
            ``out_parts[-1]``. This helper exists for clarity."""
            pass

        for i in ids:
            tok = self.id_to_token(int(i))
            if skip_special and tok in specials:
                if tok == "<sep>":
                    # Encode ``<sep>`` as a literal space (unless pending
                    # leading op is waiting, in which case the space has
                    # already been consumed by that op).
                    if pending_leading_op is None:
                        if out_parts and not out_parts[-1].endswith(" "):
                            out_parts.append(" ")
                continue
            if tok in glue_ops:
                # Decide leading vs trailing based on what came immediately
                # before: a trailing space means leading.
                if out_parts and out_parts[-1].endswith(" "):
                    # Leading operator: append the operator to the trailing
                    # space (so the space is preserved). The next piece
                    # will be glued without an intervening space.
                    out_parts[-1] = out_parts[-1] + tok
                    pending_leading_op = tok
                else:
                    # Trailing operator: glue to previous piece.
                    if out_parts:
                        out_parts[-1] = out_parts[-1] + tok
                    else:
                        out_parts.append(tok)
                    pending_leading_op = None
            else:
                if pending_leading_op is not None:
                    # Glue to the operator (no leading space).
                    out_parts[-1] = out_parts[-1] + tok
                    pending_leading_op = None
                else:
                    if out_parts and not out_parts[-1].endswith(" "):
                        out_parts.append(" ")
                    out_parts.append(tok)
        _flush_pending_op_to_glue()
        s = "".join(out_parts).strip()
        return s

    # ----- structured context encoding ------------------------------------

    def encode_context(self, ctx: "HiveContext") -> List[int]:
        """Encode a :class:`HiveContext` into a flat id list.

        The format mirrors the canonical example from the adaptation plan::

            <bos> <state> Base+MLP ; InProgress ; White [ 12 ]
            <features> white_queen_in_play yes ; ...
            <history> wB1 ; bB1 wB1/ ; ...
            <legal> wA1 /bQ ; wG2 bS1- ; pass
            <move> <mask> ... <eos>
        """
        # Populate the per-game-type piece-availability mask.
        if ctx.illegal_piece_ids is None:
            ctx.illegal_piece_ids = self.illegal_piece_ids_for_game_type(
                ctx.game_type)
        toks: List[str] = ["<bos>", "<state>", ctx.game_type, ";",
                           ctx.board_state, ";", ctx.current_color,
                           "[", f"n{max(0, min(63, ctx.current_turn))}", "]"]
        if ctx.features:
            toks.append("<features>")
            for k, v in ctx.features:
                toks.append(k)
                toks.append(v)
                toks.append(";")
            # Trim trailing ';'
            if toks[-1] == ";":
                toks.pop()
        if ctx.history:
            toks.append("<history>")
            for mv in ctx.history:
                toks.append(mv)
                toks.append(";")
            if toks[-1] == ";":
                toks.pop()
        if ctx.legal_moves:
            toks.append("<legal>")
            for mv in ctx.legal_moves:
                toks.append(mv)
                toks.append(";")
            if toks[-1] == ";":
                toks.pop()
        if ctx.target_move is not None:
            toks.append("<move>")
            toks.append(ctx.target_move)
        return self.encode_text(" ".join(toks), add_bos=False, add_eos=ctx.target_move is None)

    # ----- validation helpers --------------------------------------------

    def assert_roundtrip(self, text: str) -> None:
        """Assert that ``decode(encode(text))`` recovers ``text`` (modulo whitespace)."""
        ids = self.encode_text(text, add_bos=False, add_eos=False)
        out = self.decode(ids, skip_special=True)
        # Normalize whitespace
        norm = " ".join(text.split())
        norm_out = " ".join(out.split())
        if norm != norm_out:
            raise AssertionError(f"roundtrip failed:\n  in : {norm!r}\n  out: {norm_out!r}")


@dataclass
class HiveContext:
    """Structured Hive context for the model.

    Attributes
    ----------
    game_type : str
        Game type token (e.g. ``"Base+MLP"``).
    board_state : str
        Board state token (e.g. ``"InProgress"``).
    current_color : str
        ``"White"`` or ``"Black"``.
    current_turn : int
        1-indexed turn number (will be encoded via ``n<turn>``).
    features : list of (key, value) pairs
        Auxiliary features (e.g. ``("white_queen_in_play", "yes")``).
    history : list of move strings
        Mzinga-canonical move strings played so far.
    legal_moves : list of move strings
        All legal moves for the current position (from Mzinga).
    target_move : str, optional
        The move the model is being trained to predict.
    """

    game_type: str
    board_state: str
    current_color: str
    current_turn: int
    features: List[Tuple[str, str]] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    legal_moves: List[str] = field(default_factory=list)
    target_move: Optional[str] = None
    # Per-game-type piece availability mask (NEXT_STEPS 1.4).
    # Set by the context builder for the model to use.
    illegal_piece_ids: Optional[set] = None


def build_default_tokenizer(cfg: Optional[HiveDiffusionConfig] = None) -> HiveTokenizer:
    """Build a tokenizer sized for the given config (or a default smoke config)."""
    cfg = cfg or HiveDiffusionConfig()
    return HiveTokenizer(cfg=cfg)