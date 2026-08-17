from __future__ import annotations

from dataclasses import dataclass

from mzinga.core.enums import PieceName
from mzinga.core.position import NULL_POSITION, Position, pack_position


PASS_STRING = "pass"

_DIRECTION_SEPARATORS = frozenset(["/", "\\", "-"])


@dataclass(frozen=True)
class Move:
    piece_name: PieceName
    source: Position
    destination: Position

    @staticmethod
    def _try_parse_piece_name(s: str) -> PieceName:
        for member in PieceName:
            if member.name == s:
                return member
        s_lower = s.lower()
        for member in PieceName:
            if member.name.lower() == s_lower:
                return member
        return PieceName.INVALID

    @staticmethod
    def build_move_string(
        is_pass: bool,
        start_piece: PieceName,
        before_separator: str,
        end_piece: PieceName,
        after_separator: str,
    ) -> str:
        if is_pass:
            return PASS_STRING

        result = start_piece.name
        if end_piece != PieceName.INVALID:
            result += " "
            if before_separator:
                result += f"{before_separator}{end_piece.name}"
            elif after_separator:
                result += f"{end_piece.name}{after_separator}"
            else:
                result += end_piece.name
        return result

    @staticmethod
    def try_normalize_move_string(
        move_string: str,
    ) -> tuple[bool, PieceName, str, PieceName, str] | None:
        piece1_chars: list[str] = []
        piece2_chars: list[str] = []
        before_separator = ""
        after_separator = ""

        items_found = 0
        i = 0
        while i < len(move_string):
            ch = move_string[i]
            if items_found == 0 and ch != " ":
                piece1_chars.append(ch)
                items_found = 1
            elif items_found == 1:
                if ch != " ":
                    piece1_chars.append(ch)
                else:
                    items_found = 2
            elif items_found == 2:
                if ch != " ":
                    if ch in _DIRECTION_SEPARATORS:
                        before_separator = ch
                    else:
                        piece2_chars.append(ch)
                    items_found = 3
            elif items_found == 3:
                if ch != " ":
                    if ch in _DIRECTION_SEPARATORS:
                        after_separator = ch
                        break
                    else:
                        piece2_chars.append(ch)
                else:
                    break
            i += 1

        piece1_str = "".join(piece1_chars)

        if piece1_str.lower() == PASS_STRING:
            return (True, PieceName.INVALID, "", PieceName.INVALID, "")

        start_piece = Move._try_parse_piece_name(piece1_str)
        if start_piece == PieceName.INVALID:
            return None

        piece2_str = "".join(piece2_chars)

        if not piece2_str and not before_separator and not after_separator:
            return (False, start_piece, "", PieceName.INVALID, "")

        if piece2_str:
            end_piece = Move._try_parse_piece_name(piece2_str)
            if end_piece != PieceName.INVALID:
                return (False, start_piece, before_separator, end_piece, after_separator)

        return None

    @staticmethod
    def try_normalize_move_string_full(move_string: str) -> str | None:
        result = Move.try_normalize_move_string(move_string)
        if result is not None:
            is_pass, start_piece, before_separator, end_piece, after_separator = result
            return Move.build_move_string(
                is_pass, start_piece, before_separator, end_piece, after_separator
            )
        return None


PASS_MOVE = Move(PieceName.INVALID, NULL_POSITION, NULL_POSITION)
