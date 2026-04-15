"""Albrecht tournament agent — expectiminimax + belief filter + D5 hardening."""

import gc
import os
from collections.abc import Callable
from typing import Tuple

from game import board, move, enums

from .t_precompute import Precomputed
from .belief import RatBelief
from .search import Searcher

_DEBUG = os.environ.get("ALBRECHT_DEBUG") == "1"

_TIME_RESERVE = 15.0
_PANIC_TIME = 20.0
_PANIC_BUDGET = 0.25
_GC_EVERY = 4

try:
    import resource as _resource
except ImportError:
    _resource = None


class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        # Precompute transition-matrix-derived data
        self.pre = Precomputed(transition_matrix)
        self.belief = RatBelief(self.pre)
        self.searcher = Searcher()
        self.turn = 0

    def commentate(self):
        return "gg, the rat never stood a chance"

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        self.turn += 1
        try:
            return self._play_impl(board, sensor_data, time_left)
        except Exception as e:
            if _DEBUG:
                print(f"[Albrecht T{self.turn}] top-level exception: {e!r}")
            return self._safe_fallback(board)

    def _play_impl(self, board, sensor_data, time_left):
        noise, dist = sensor_data

        # --- Belief update pipeline (best-effort; on failure search runs without belief) ---
        belief_for_search = self.belief
        try:
            our_search_loc, our_search_hit = board.player_search
            self.belief.update_search(our_search_loc, our_search_hit)
            self.belief.predict()
            opp_search_loc, opp_search_hit = board.opponent_search
            self.belief.update_search(opp_search_loc, opp_search_hit)
            self.belief.predict()
            worker_pos = board.player_worker.get_location()
            self.belief.update_sensor(noise, dist, worker_pos, board)
        except Exception as e:
            if _DEBUG:
                print(f"[Albrecht T{self.turn}] belief exception: {e!r}")
            belief_for_search = None

        if _DEBUG and belief_for_search is not None:
            bsum = self.belief.b.sum()
            assert abs(bsum - 1.0) < 1e-3, f"Turn {self.turn}: belief sum = {bsum}"
            print(f"[Albrecht T{self.turn}] belief argmax={self.belief.argmax()} "
                  f"p={self.belief.max_prob():.3f} ev_search={self.belief.ev_best_search():.2f}")

        # --- Time budget with panic mode ---
        remaining = time_left()
        turns_remaining = board.player_worker.turns_left
        if turns_remaining <= 0:
            turns_remaining = 1

        if remaining < _PANIC_TIME:
            budget = _PANIC_BUDGET
            panic = True
        else:
            budget = max(0.5, (remaining - _TIME_RESERVE) / turns_remaining)
            panic = False

        # --- Search with layered fallback ---
        best_move = None
        try:
            best_move = self.searcher.search(board, belief_for_search, budget)
        except Exception as e:
            if _DEBUG:
                print(f"[Albrecht T{self.turn}] search exception: {e!r}")
            # Try shallowest move from previous TT / candidate ordering
            try:
                cand = board.get_valid_moves(exclude_search=True)
                if cand:
                    best_move = cand[0]
            except Exception:
                pass

        if best_move is None:
            best_move = self._safe_fallback(board)

        # --- Periodic GC and RSS log to keep memory tame ---
        if self.turn % _GC_EVERY == 0:
            try:
                gc.collect()
            except Exception:
                pass
            if _DEBUG and _resource is not None:
                rss_kb = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
                print(f"[Albrecht T{self.turn}] rss_max={rss_kb} kb tt_size={len(self.searcher.tt)}")

        if _DEBUG:
            print(f"[Albrecht T{self.turn}] budget={budget:.2f}s panic={panic} "
                  f"remaining={remaining:.1f}s turns_left={turns_remaining} "
                  f"move={best_move}")

        return best_move

    def _safe_fallback(self, board):
        try:
            moves = board.get_valid_moves()
            if moves:
                return moves[0]
        except Exception:
            pass
        return move.Move.plain(enums.Direction.UP)
