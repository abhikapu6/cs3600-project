"""Albrecht tournament agent — skeleton + belief filter (D1)."""

import os
import random
from collections.abc import Callable
from typing import Tuple

from game import board, move, enums

from .t_precompute import Precomputed
from .belief import RatBelief

_DEBUG = os.environ.get("ALBRECHT_DEBUG") == "1"


class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        # Precompute transition-matrix-derived data
        self.pre = Precomputed(transition_matrix)
        self.belief = RatBelief(self.pre)
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
        # Between our turns, 2 rat moves happen (one on opponent's turn, one on ours).
        # Timeline: our turn T → rat.move (T+1) → opponent plays (may search) →
        #           rat.move (T+2) → we get sensor → our turn T+2.
        #
        # Correct order:
        #   1. Fold our own search result from turn T (we didn't know hit/miss then)
        #   2. predict() for T+1 rat move
        #   3. Fold opponent's search from T+1
        #   4. predict() for T+2 rat move
        #   5. Sensor update from T+2's sample

        # 1. Fold our own search result from 2 turns ago
        our_search_loc, our_search_hit = board.player_search
        self.belief.update_search(our_search_loc, our_search_hit)

        # 2. Predict rat move during opponent's turn (T+1)
        self.belief.predict()

        # 3. Fold opponent's search from their last turn (T+1)
        opp_search_loc, opp_search_hit = board.opponent_search
        self.belief.update_search(opp_search_loc, opp_search_hit)

        # 4. Predict rat move at start of our turn (T+2)
        self.belief.predict()

        # 5. Sensor update (noise + distance measured at T+2 after rat moved)
        worker_pos = board.player_worker.get_location()
        self.belief.update_sensor(noise, dist, worker_pos, board)

        if _DEBUG:
            bsum = self.belief.b.sum()
            assert abs(bsum - 1.0) < 1e-3, f"Turn {self.turn}: belief sum = {bsum}"
            best_pos = self.belief.argmax()
            best_p = self.belief.max_prob()
            print(f"[Albrecht T{self.turn}] belief argmax={best_pos} p={best_p:.3f} ev_search={self.belief.ev_best_search():.2f}")

        # --- Move selection (D1: random valid move) ---
        moves = board.get_valid_moves()
        return random.choice(moves)
