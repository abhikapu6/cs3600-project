"""Iterative-deepening negamax with alpha-beta pruning + Zobrist TT (D3)."""

import os
import time

from game.enums import MoveType, CARPET_POINTS_TABLE

from .eval import evaluate
from .zobrist import board_hash

_DEBUG = os.environ.get("ALBRECHT_DEBUG") == "1"

# TT entry flags
_EXACT = 0
_LOWER = 1  # failed high (beta cutoff) — stored value is a lower bound
_UPPER = 2  # failed low — stored value is an upper bound

# TT size cap
_TT_MAX = 1 << 18  # 262144 entries


class _TTEntry:
    __slots__ = ('key', 'depth', 'value', 'flag', 'best_move')

    def __init__(self, key, depth, value, flag, best_move):
        self.key = key
        self.depth = depth
        self.value = value
        self.flag = flag
        self.best_move = best_move


class _Timeout(Exception):
    pass


class Searcher:
    """Expectiminimax searcher using negamax formulation with TT."""

    def __init__(self):
        self.nodes = 0
        self.best_move = None
        self.max_depth_completed = 0
        self.tt = {}
        self.tt_hits = 0

    def search(self, board, belief, time_budget):
        """Run iterative-deepening search and return the best move."""
        self.start_time = time.monotonic()
        self.deadline = self.start_time + 0.9 * time_budget
        self.nodes = 0
        self.tt_hits = 0
        self.max_depth_completed = 0

        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
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

            elapsed = time.monotonic() - self.start_time
            if elapsed >= 0.5 * (self.deadline - self.start_time):
                break

        # Evict TT if too large
        if len(self.tt) > _TT_MAX:
            self._evict_tt()

        if _DEBUG:
            elapsed = time.monotonic() - self.start_time
            print(f"[Albrecht search] depth={self.max_depth_completed} "
                  f"nodes={self.nodes} tt_hits={self.tt_hits} "
                  f"tt_size={len(self.tt)} time={elapsed:.3f}s "
                  f"move={self.best_move}")

        return self.best_move

    def _evict_tt(self):
        """Evict low-depth entries when TT exceeds size cap."""
        if len(self.tt) <= _TT_MAX:
            return
        # Keep entries with highest depth
        entries = list(self.tt.items())
        entries.sort(key=lambda kv: kv[1].depth, reverse=True)
        self.tt = dict(entries[:_TT_MAX * 3 // 4])

    def _check_time(self):
        if time.monotonic() >= self.deadline:
            raise _Timeout

    def _root_search(self, board, moves, max_depth):
        """Search from root at a fixed depth. Returns (score, best_move)."""
        alpha = float('-inf')
        beta = float('inf')
        best_move = None
        best_score = float('-inf')

        # Order moves: TT best first, then by type priority
        h = board_hash(board)
        tt_entry = self.tt.get(h)
        tt_best = tt_entry.best_move if tt_entry and tt_entry.key == h else None
        ordered = self._order_moves(moves, tt_best)

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

        # Store root in TT
        if best_move is not None:
            self.tt[h] = _TTEntry(h, max_depth, best_score, _EXACT, best_move)

        return best_score, best_move

    def _negamax(self, board, depth, alpha, beta):
        """Negamax with alpha-beta and TT. Returns score from current player's perspective."""
        self.nodes += 1
        self._check_time()

        if board.is_game_over():
            return self._terminal_score(board)

        if depth <= 0:
            return evaluate(board)

        # TT probe
        h = board_hash(board)
        tt_entry = self.tt.get(h)
        tt_best = None
        if tt_entry and tt_entry.key == h:
            if tt_entry.depth >= depth:
                self.tt_hits += 1
                if tt_entry.flag == _EXACT:
                    return tt_entry.value
                elif tt_entry.flag == _LOWER:
                    if tt_entry.value >= beta:
                        return tt_entry.value
                    if tt_entry.value > alpha:
                        alpha = tt_entry.value
                elif tt_entry.flag == _UPPER:
                    if tt_entry.value <= alpha:
                        return tt_entry.value
                    if tt_entry.value < beta:
                        beta = tt_entry.value
            tt_best = tt_entry.best_move

        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            return evaluate(board)

        ordered = self._order_moves(moves, tt_best)
        best_score = float('-inf')
        best_move = None
        orig_alpha = alpha

        for move in ordered:
            child = board.forecast_move(move)
            if child is None:
                continue
            child.reverse_perspective()

            score = -self._negamax(child, depth - 1, -beta, -alpha)

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break  # beta cutoff

        # Store in TT (replace-by-depth)
        if best_score > float('-inf'):
            if best_score <= orig_alpha:
                flag = _UPPER
            elif best_score >= beta:
                flag = _LOWER
            else:
                flag = _EXACT
            existing = self.tt.get(h)
            if existing is None or existing.depth <= depth:
                self.tt[h] = _TTEntry(h, depth, best_score, flag, best_move)

        return best_score if best_score > float('-inf') else evaluate(board)

    def _terminal_score(self, board):
        """Score for a finished game from player_worker's perspective."""
        my_pts = board.player_worker.get_points()
        opp_pts = board.opponent_worker.get_points()
        diff = my_pts - opp_pts
        if diff > 0:
            return 1000.0 + diff
        elif diff < 0:
            return -1000.0 + diff
        else:
            return 0.0

    @staticmethod
    def _moves_match(a, b):
        """Check if two Move objects represent the same move."""
        if a.move_type != b.move_type:
            return False
        if a.move_type == MoveType.CARPET:
            return a.direction == b.direction and a.roll_length == b.roll_length
        if a.move_type == MoveType.SEARCH:
            return a.search_loc == b.search_loc
        return a.direction == b.direction  # PLAIN or PRIME

    @staticmethod
    def _order_moves(moves, tt_best=None):
        """Order moves: TT best → carpet (desc) → prime → plain → search."""
        if tt_best is not None:
            # Put TT best move first, rest sorted by type priority
            first = None
            rest = []
            for m in moves:
                if first is None and Searcher._moves_match(m, tt_best):
                    first = m
                else:
                    rest.append(m)

            def priority(m):
                if m.move_type == MoveType.CARPET:
                    return (0, -CARPET_POINTS_TABLE.get(m.roll_length, 0))
                elif m.move_type == MoveType.PRIME:
                    return (1, 0)
                elif m.move_type == MoveType.PLAIN:
                    return (2, 0)
                else:
                    return (3, 0)

            rest.sort(key=priority)
            if first is not None:
                return [first] + rest
            return rest

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
