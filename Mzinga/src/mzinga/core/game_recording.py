from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TextIO

from mzinga.core.board import Board
from mzinga.core.enums import GameType


@dataclass
class GameMetadata:
    event: str = "?"
    site: str = "?"
    date: str = "?"
    round: str = "?"
    white: str = "?"
    black: str = "?"
    result: str = "*"
    game_string: str = ""
    white_elo: str = ""
    black_elo: str = ""
    eco: str = ""
    opening: str = ""
    optional_tags: dict[str, str] = field(default_factory=dict)

    def get_tag(self, key: str) -> Optional[str]:
        mapping = {
            "Event": self.event,
            "Site": self.site,
            "Date": self.date,
            "Round": self.round,
            "White": self.white,
            "Black": self.black,
            "Result": self.result,
            "GameString": self.game_string,
            "WhiteElo": self.white_elo,
            "BlackElo": self.black_elo,
            "ECO": self.eco,
            "Opening": self.opening,
        }
        if key in mapping:
            return mapping[key]
        return self.optional_tags.get(key)

    def set_tag(self, key: str, value: str) -> None:
        value = value.replace('"', "").strip()
        if key == "Event":
            self.event = value
        elif key == "Site":
            self.site = value
        elif key == "Date":
            self.date = value
        elif key == "Round":
            self.round = value
        elif key == "White":
            self.white = value
        elif key == "Black":
            self.black = value
        elif key == "Result":
            self.result = value
        elif key == "GameString":
            self.game_string = value
        elif key == "WhiteElo":
            self.white_elo = value
        elif key == "BlackElo":
            self.black_elo = value
        elif key == "ECO":
            self.eco = value
        elif key == "Opening":
            self.opening = value
        else:
            self.optional_tags[key] = value


class GameRecording:
    def __init__(self, metadata: Optional[GameMetadata] = None):
        self.metadata = metadata or GameMetadata()
        self.moves: list[str] = []
        self.board: Optional[Board] = None

    def load_pgn(self, file: TextIO) -> None:
        tags = {}
        movelines = []
        for line in file:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                inner = line[1:-1]
                space_idx = inner.find(" ")
                if space_idx > 0:
                    tag_name = inner[:space_idx].strip()
                    tag_value = inner[space_idx + 1:].strip().strip('"')
                    tags[tag_name] = tag_value
            elif line and not line.startswith("{") and not line.startswith(";"):
                movelines.append(line)

        self.metadata = GameMetadata(
            event=tags.get("Event", "?"),
            site=tags.get("Site", "?"),
            date=tags.get("Date", "?"),
            round=tags.get("Round", "?"),
            white=tags.get("White", "?"),
            black=tags.get("Black", "?"),
            result=tags.get("Result", "*"),
            game_string=tags.get("GameString", ""),
            white_elo=tags.get("WhiteElo", ""),
            black_elo=tags.get("BlackElo", ""),
            eco=tags.get("ECO", ""),
            opening=tags.get("Opening", ""),
        )

        move_text = " ".join(movelines)
        self.moves = self._parse_pgn_moves(move_text)

        if self.metadata.game_string:
            try:
                self.board = Board.parse_game_string(self.metadata.game_string, trusted_play=True)
            except Exception:
                self.board = self._replay_moves(self.moves)
        else:
            self.board = self._replay_moves(self.moves)

    def load_sgf(self, file: TextIO) -> None:
        content = file.read()
        import re

        comments = re.findall(r'C\[([^\]]*)\]', content)
        game_string = None
        for c in comments:
            if ";" in c:
                game_string = c.strip()
                break

        if game_string:
            self.metadata.game_string = game_string
            try:
                self.board = Board.parse_game_string(game_string, trusted_play=True)
            except Exception:
                pass

        properties = re.findall(r'([A-Z]+)\[([^\]]*)\]', content)
        self.metadata.white = self._get_sgf_prop(properties, "PW", "?")
        self.metadata.black = self._get_sgf_prop(properties, "PB", "?")
        self.metadata.date = self._get_sgf_prop(properties, "DT", "?")
        self.metadata.result = self._get_sgf_prop(properties, "RE", "*")
        self.metadata.event = self._get_sgf_prop(properties, "EV", "?")
        self.metadata.site = self._get_sgf_prop(properties, "PC", "?")

    def save_pgn(self, file: TextIO) -> None:
        self._write_pgn_tag(file, "Event", self.metadata.event)
        self._write_pgn_tag(file, "Site", self.metadata.site)
        self._write_pgn_tag(file, "Date", self.metadata.date)
        self._write_pgn_tag(file, "Round", self.metadata.round)
        self._write_pgn_tag(file, "White", self.metadata.white)
        self._write_pgn_tag(file, "Black", self.metadata.black)
        self._write_pgn_tag(file, "Result", self.metadata.result)
        if self.metadata.game_string:
            self._write_pgn_tag(file, "GameString", self.metadata.game_string)
        if self.metadata.white_elo:
            self._write_pgn_tag(file, "WhiteElo", self.metadata.white_elo)
        if self.metadata.black_elo:
            self._write_pgn_tag(file, "BlackElo", self.metadata.black_elo)
        file.write("\n")

        if self.board:
            moves = []
            for item in self.board.board_history:
                moves.append(item.move_string)
            count = 1
            line = ""
            for move in moves:
                if line:
                    line = ""
                file.write("{0}. {1}\n".format(count, move))
                count += 1
            file.write(self.metadata.result + "\n")

    @staticmethod
    def _write_pgn_tag(file: TextIO, name: str, value: str) -> None:
        file.write(f'[{name} "{value}"]\n')

    @staticmethod
    def _parse_pgn_moves(text: str) -> list[str]:
        import re
        text = re.sub(r"\d+\.\s*", "", text)
        text = re.sub(r"\s*(1-0|0-1|1/2-1/2|\*)\s*$", "", text)
        result = []
        for t in text.split():
            if t not in ("1-0", "0-1", "1/2-1/2", "*"):
                result.append(t)
        return result

    @staticmethod
    def _get_sgf_prop(props: list[tuple[str, str]], name: str, default: str = "") -> str:
        for prop_name, prop_value in props:
            if prop_name == name:
                return prop_value
        return default

    @staticmethod
    def _replay_moves(moves: list[str]) -> Optional[Board]:
        try:
            b = Board()
            for move_str in moves:
                move, parsed = b.parse_move(move_str)
                if move is None:
                    return None
                b.trusted_play(move, parsed)
            return b
        except Exception:
            return None
