from mzinga.core.enums import BoardState, PieceName
from mzinga.core.piece_metrics import PieceMetrics

class BoardMetrics:
    def __init__(self):
        self.board_state: BoardState = BoardState.NotStarted
        self.pieces_in_play: int = 0
        self.pieces_in_hand: int = 0
        self._piece_metrics: list[PieceMetrics] = [PieceMetrics() for _ in range(int(PieceName.NumPieceNames))]
    
    def __getitem__(self, piece_name: PieceName) -> PieceMetrics:
        return self._piece_metrics[int(piece_name)]
    
    def reset(self):
        self.board_state = BoardState.NotStarted
        self.pieces_in_play = 0
        self.pieces_in_hand = 0
        for pm in self._piece_metrics:
            pm.reset()
