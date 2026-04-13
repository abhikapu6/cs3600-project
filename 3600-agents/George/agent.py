"""George stub — greedy prime-then-carpet bot (reference baseline ~70% grade)."""

from collections.abc import Callable
from typing import Tuple

from game import board, move, enums
from game.enums import MoveType, CARPET_POINTS_TABLE, Direction


class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        pass

    def commentate(self):
        return ""

    def play(self, board: board.Board, sensor_data: Tuple, time_left: Callable):
        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            return move.Move.plain(Direction.UP)

        # 1. Best carpet move (highest points)
        best_carpet = None
        best_carpet_pts = -999
        for m in moves:
            if m.move_type == MoveType.CARPET:
                pts = CARPET_POINTS_TABLE.get(m.roll_length, -999)
                if pts > best_carpet_pts:
                    best_carpet_pts = pts
                    best_carpet = m
        if best_carpet and best_carpet_pts >= 2:
            return best_carpet

        # 2. Prime if possible
        for m in moves:
            if m.move_type == MoveType.PRIME:
                return m

        # 3. Any plain move
        for m in moves:
            if m.move_type == MoveType.PLAIN:
                return m

        return moves[0]
