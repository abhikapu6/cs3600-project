"""Iterative-deepening negamax with alpha-beta pruning (D2: no TT yet)."""

import os
import time

from game.enums import MoveType, CARPET_POINTS_TABLE

from .eval import evaluate

_DEBUG = os.environ.get("ALBRECHT_DEBUG") == "1"


class _Timeout(Exception):
    pass


class Searcher:
    """Expectiminimax searcher using negamax formulation with perspective reversal."""

    def __init__(self):
        self.nodes = 0
        self.best_move = None
        self.max_depth_completed = 0

    def search(self, board, belief, time_budget):
        """Run iterative-deepening search and return the best move.

        Args:
            board: Board object (from our perspective — player_worker = us)
            belief: RatBelief (unused in D2, will be used for chance nodes in D4)
            time_budget: seconds allocated for this turn
        Returns:
            best Move found
        """
        self.start_time = time.monotonic()
        self.deadline = self.start_time + 0.9 * time_budget
        self.nodes = 0
        self.max_depth_completed = 0

        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            # Fallback: include search moves
            moves = board.get_valid_moves(exclude_search=False)
        if not moves:
            from game.move import Move
            from game.enums import Direction
            return Move.plain(Direction.UP)

        self.best_move = moves[0]

        for depth in range(1, 30):
            try:
                score, move = self._root_search(board, moves, depth)
                if move is not None:
                    self.best_move = move
                self.max_depth_completed = depth
            except _Timeout:
                break

            # Check if we have time for the next depth
            elapsed = time.monotonic() - self.start_time
            if elapsed >= 0.5 * (self.deadline - self.start_time):
                break

        if _DEBUG:
            elapsed = time.monotonic() - self.start_time
            print(f"[Albrecht search] depth={self.max_depth_completed} "
                  f"nodes={self.nodes} time={elapsed:.3f}s move={self.best_move}")

        return self.best_move

    def _check_time(self):
        if time.monotonic() >= self.deadline:
            raise _Timeout

    def _root_search(self, board, moves, max_depth):
        """Search from root at a fixed depth. Returns (score, best_move)."""
        alpha = float('-inf')
        beta = float('inf')
        best_move = None
        best_score = float('-inf')

        ordered = self._order_moves(moves)

        for move in ordered:
            self._check_time()
            child = board.forecast_move(move)
            if child is None:
                continue
            child.reverse_perspective()

            score = -self._negamax(child, max_depth - 1, -beta, -alpha)

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score

        return best_score, best_move

    def _negamax(self, board, depth, alpha, beta):
        """Negamax with alpha-beta. Returns score from current player's perspective."""
        self.nodes += 1
        self._check_time()

        # Terminal check
        if board.is_game_over():
            return self._terminal_score(board)

        if depth <= 0:
            return evaluate(board)

        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            return evaluate(board)

        ordered = self._order_moves(moves)
        best_score = float('-inf')

        for move in ordered:
            child = board.forecast_move(move)
            if child is None:
                continue
            child.reverse_perspective()

            score = -self._negamax(child, depth - 1, -beta, -alpha)

            if score > best_score:
                best_score = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break  # beta cutoff

        return best_score if best_score > float('-inf') else evaluate(board)

    def _terminal_score(self, board):
        """Score for a finished game from player_worker's perspective."""
        my_pts = board.player_worker.get_points()
        opp_pts = board.opponent_worker.get_points()
        diff = my_pts - opp_pts
        # Large bonus/penalty for win/loss to guide search toward winning lines
        if diff > 0:
            return 1000.0 + diff
        elif diff < 0:
            return -1000.0 + diff
        else:
            return 0.0

    @staticmethod
    def _order_moves(moves):
        """Order moves: carpet (desc by score) > prime > plain.

        Good move ordering improves alpha-beta pruning dramatically.
        """
        def priority(m):
            if m.move_type == MoveType.CARPET:
                return (0, -CARPET_POINTS_TABLE.get(m.roll_length, 0))
            elif m.move_type == MoveType.PRIME:
                return (1, 0)
            elif m.move_type == MoveType.PLAIN:
                return (2, 0)
            else:
                return (3, 0)

        return sorted(moves, key=priority)
