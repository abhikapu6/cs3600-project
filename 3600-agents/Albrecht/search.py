"""Iterative-deepening negamax with alpha-beta + Zobrist TT + belief-integrated SEARCH (D4)."""

import os
import time

from game.enums import MoveType, CARPET_POINTS_TABLE, BOARD_SIZE, RAT_BONUS, RAT_PENALTY
from game.move import Move

from .eval import evaluate
from .zobrist import board_hash

_DEBUG = os.environ.get("ALBRECHT_DEBUG") == "1"

_EXACT = 0
_LOWER = 1
_UPPER = 2

_TT_MAX = 1 << 18

# Max SEARCH chance-node candidates to expand per node
_SEARCH_TOPK = 3
# Also expand cells with belief probability above this threshold.
# True myopic break-even is p = 2/6 = 0.333, but a miss concentrates belief
# and enables high-EV searches next turn — so we accept slightly negative
# myopic EV to capture that information value.
_SEARCH_BELIEF_THRESHOLD = 0.20
# Accept search candidates with EV above this floor (was 0.0).
_SEARCH_EV_FLOOR = -1.0


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


def _idx(loc):
    return loc[1] * BOARD_SIZE + loc[0]


class Searcher:
    """Belief-integrated expectiminimax searcher (negamax form)."""

    def __init__(self):
        self.nodes = 0
        self.best_move = None
        self.max_depth_completed = 0
        self.tt = {}
        self.tt_hits = 0

    def search(self, board, belief, time_budget):
        self.start_time = time.monotonic()
        self.deadline = self.start_time + 0.9 * time_budget
        self.nodes = 0
        self.tt_hits = 0
        self.max_depth_completed = 0

        moves = self._candidate_moves(board, belief)
        if not moves:
            from game.enums import Direction
            return Move.plain(Direction.UP)

        self.best_move = moves[0]

        for depth in range(1, 30):
            try:
                score, best_move = self._root_search(board, belief, moves, depth)
                if best_move is not None:
                    self.best_move = best_move
                self.max_depth_completed = depth
            except _Timeout:
                break
            except Exception as e:
                if _DEBUG:
                    print(f"[Albrecht search] depth {depth} exception: {e!r}")
                break

            elapsed = time.monotonic() - self.start_time
            if elapsed >= 0.5 * (self.deadline - self.start_time):
                break
            if len(self.tt) > _TT_MAX:
                self._evict_tt()

        if len(self.tt) > _TT_MAX:
            self._evict_tt()

        if _DEBUG:
            elapsed = time.monotonic() - self.start_time
            print(f"[Albrecht search] depth={self.max_depth_completed} "
                  f"nodes={self.nodes} tt_hits={self.tt_hits} "
                  f"tt_size={len(self.tt)} time={elapsed:.3f}s "
                  f"move={self.best_move}")

        return self.best_move

    def _candidate_moves(self, board, belief):
        """All non-search valid moves + top-K belief-weighted search candidates."""
        non_search = board.get_valid_moves(exclude_search=True)
        search_moves = self._top_search_moves(board, belief)
        return non_search + search_moves

    def _top_search_moves(self, board, belief):
        """Pick up to top-K search cells by belief probability plus any above threshold."""
        if belief is None:
            return []
        b = belief.b
        # Pair (prob, idx) and pick highest
        idxs = b.argsort()[::-1]
        picks = []
        threshold_hits = []
        seen = set()
        for i in idxs[:_SEARCH_TOPK]:
            i = int(i)
            if i in seen:
                continue
            seen.add(i)
            loc = (i % BOARD_SIZE, i // BOARD_SIZE)
            picks.append(Move.search(loc))
        for i in range(BOARD_SIZE * BOARD_SIZE):
            if i in seen:
                continue
            if float(b[i]) >= _SEARCH_BELIEF_THRESHOLD:
                loc = (i % BOARD_SIZE, i // BOARD_SIZE)
                threshold_hits.append(Move.search(loc))
        # Only keep searches with positive EV (otherwise skip to save time)
        filtered = [m for m in picks + threshold_hits
                    if 6.0 * float(b[_idx(m.search_loc)]) - 2.0 > _SEARCH_EV_FLOOR]
        return filtered

    def _evict_tt(self):
        if len(self.tt) <= _TT_MAX:
            return
        entries = list(self.tt.items())
        entries.sort(key=lambda kv: kv[1].depth, reverse=True)
        self.tt = dict(entries[:_TT_MAX * 3 // 4])

    def _check_time(self):
        if time.monotonic() >= self.deadline:
            raise _Timeout

    def _root_search(self, board, belief, moves, max_depth):
        alpha = float('-inf')
        beta = float('inf')
        best_move = None
        best_score = float('-inf')

        h = board_hash(board)
        tt_entry = self.tt.get(h)
        tt_best = tt_entry.best_move if tt_entry and tt_entry.key == h else None
        ordered = self._order_moves(moves, tt_best, belief)

        for mv in ordered:
            self._check_time()
            score = self._score_child(board, belief, mv, max_depth - 1, alpha, beta)
            if score is None:
                continue

            if score > best_score:
                best_score = score
                best_move = mv
            if score > alpha:
                alpha = score

        if best_move is not None:
            self.tt[h] = _TTEntry(h, max_depth, best_score, _EXACT, best_move)

        return best_score, best_move

    def _score_child(self, board, belief, mv, depth, alpha, beta):
        """Apply move and return value from current side's perspective."""
        if mv.move_type == MoveType.SEARCH:
            # Chance node: handle hit/miss branches.
            loc = mv.search_loc
            p = float(belief.b[_idx(loc)]) if belief is not None else 0.0
            child = board.forecast_move(mv)
            if child is None:
                return None
            child.reverse_perspective()

            # Hit branch: rat caught, respawn distribution, +RAT_BONUS
            belief_hit = belief.clone() if belief is not None else None
            if belief_hit is not None:
                belief_hit.b = belief_hit.pre.spawn_dist.copy()
                belief_hit.predict()

            # Miss branch: zero out belief at loc, renormalize, -RAT_PENALTY
            belief_miss = belief.clone() if belief is not None else None
            if belief_miss is not None:
                belief_miss.b[_idx(loc)] = 0.0
                s = belief_miss.b.sum()
                if s > 0:
                    belief_miss.b /= s
                belief_miss.predict()

            # Wide window for chance-node children (bounds pruning deferred)
            v_hit_child = self._negamax(child, belief_hit, depth,
                                         float('-inf'), float('inf'))
            v_miss_child = self._negamax(child, belief_miss, depth,
                                          float('-inf'), float('inf'))

            # Convert child scores (child-side) to our side and add rewards.
            v_hit = RAT_BONUS + (-v_hit_child)
            v_miss = -RAT_PENALTY + (-v_miss_child)
            return p * v_hit + (1.0 - p) * v_miss

        child = board.forecast_move(mv)
        if child is None:
            return None
        child.reverse_perspective()
        child_belief = belief.clone() if belief is not None else None
        if child_belief is not None:
            child_belief.predict()
        return -self._negamax(child, child_belief, depth, -beta, -alpha)

    def _negamax(self, board, belief, depth, alpha, beta):
        self.nodes += 1
        self._check_time()

        if board.is_game_over():
            return self._terminal_score(board)

        if depth <= 0:
            return evaluate(board, belief)

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

        moves = self._candidate_moves(board, belief)
        if not moves:
            return evaluate(board, belief)

        ordered = self._order_moves(moves, tt_best, belief)
        best_score = float('-inf')
        best_move = None
        orig_alpha = alpha

        for mv in ordered:
            score = self._score_child(board, belief, mv, depth - 1, alpha, beta)
            if score is None:
                continue

            if score > best_score:
                best_score = score
                best_move = mv
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

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

        return best_score if best_score > float('-inf') else evaluate(board, belief)

    def _terminal_score(self, board):
        my_pts = board.player_worker.get_points()
        opp_pts = board.opponent_worker.get_points()
        diff = my_pts - opp_pts
        if diff > 0:
            return 1000.0 + diff
        elif diff < 0:
            return -1000.0 + diff
        return 0.0

    @staticmethod
    def _moves_match(a, b):
        if a.move_type != b.move_type:
            return False
        if a.move_type == MoveType.CARPET:
            return a.direction == b.direction and a.roll_length == b.roll_length
        if a.move_type == MoveType.SEARCH:
            return a.search_loc == b.search_loc
        return a.direction == b.direction

    @staticmethod
    def _move_priority(m, belief):
        if m.move_type == MoveType.CARPET:
            pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
            # Large carpets (4+) cash first; short carpets (2-3) defer below
            # PRIME so the search prefers building bigger chains.
            if m.roll_length >= 4:
                return (0, -pts)
            return (2, -pts)
        if m.move_type == MoveType.PRIME:
            return (1, 0)
        if m.move_type == MoveType.PLAIN:
            return (3, 0)
        # SEARCH: order by negative EV so high-EV searches come first
        if belief is not None:
            p = float(belief.b[_idx(m.search_loc)])
            return (4, -p)
        return (4, 0)

    @staticmethod
    def _order_moves(moves, tt_best, belief):
        if tt_best is not None:
            first = None
            rest = []
            for m in moves:
                if first is None and Searcher._moves_match(m, tt_best):
                    first = m
                else:
                    rest.append(m)
            rest.sort(key=lambda mv: Searcher._move_priority(mv, belief))
            if first is not None:
                return [first] + rest
            return rest
        return sorted(moves, key=lambda mv: Searcher._move_priority(mv, belief))
