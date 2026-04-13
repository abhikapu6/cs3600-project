"""Albrecht tournament agent — expectiminimax + belief filter (D2)."""

import os
from collections.abc import Callable
from typing import Tuple

from game import board, move, enums

from .t_precompute import Precomputed
from .belief import RatBelief
from .search import Searcher

_DEBUG = os.environ.get("ALBRECHT_DEBUG") == "1"

# Safety buffer in seconds to reserve from total time
_TIME_RESERVE = 15.0


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
        noise, dist = sensor_data
        self.turn += 1

        # --- Belief update pipeline ---
        # 1. Fold our own search result from last turn
        our_search_loc, our_search_hit = board.player_search
        self.belief.update_search(our_search_loc, our_search_hit)

        # 2. Predict rat move during opponent's turn
        self.belief.predict()

        # 3. Fold opponent's search from their last turn
        opp_search_loc, opp_search_hit = board.opponent_search
        self.belief.update_search(opp_search_loc, opp_search_hit)

        # 4. Predict rat move at start of our turn
        self.belief.predict()

        # 5. Sensor update
        worker_pos = board.player_worker.get_location()
        self.belief.update_sensor(noise, dist, worker_pos, board)

        if _DEBUG:
            bsum = self.belief.b.sum()
            assert abs(bsum - 1.0) < 1e-3, f"Turn {self.turn}: belief sum = {bsum}"
            best_pos = self.belief.argmax()
            best_p = self.belief.max_prob()
            print(f"[Albrecht T{self.turn}] belief argmax={best_pos} "
                  f"p={best_p:.3f} ev_search={self.belief.ev_best_search():.2f}")

        # --- Time budget ---
        remaining = time_left()
        turns_remaining = board.player_worker.turns_left
        if turns_remaining <= 0:
            turns_remaining = 1
        budget = max(0.5, (remaining - _TIME_RESERVE) / turns_remaining)

        # --- Search ---
        try:
            best_move = self.searcher.search(board, self.belief, budget)
        except Exception as e:
            if _DEBUG:
                print(f"[Albrecht] search exception: {e}")
            # Fallback to any valid move
            moves = board.get_valid_moves()
            best_move = moves[0] if moves else move.Move.plain(enums.Direction.UP)

        if _DEBUG:
            print(f"[Albrecht T{self.turn}] budget={budget:.2f}s "
                  f"remaining={remaining:.1f}s turns_left={turns_remaining} "
                  f"move={best_move}")

        return best_move
