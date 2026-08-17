class PieceMetrics:
    def __init__(self):
        self.in_play: int = 0
        self.is_pinned: int = 0
        self.is_covered: int = 0
        self.noisy_move_count: int = 0
        self.quiet_move_count: int = 0
        self.friendly_neighbor_count: int = 0
        self.enemy_neighbor_count: int = 0
    
    def reset(self):
        self.in_play = 0
        self.is_pinned = 0
        self.is_covered = 0
        self.noisy_move_count = 0
        self.quiet_move_count = 0
        self.friendly_neighbor_count = 0
        self.enemy_neighbor_count = 0
