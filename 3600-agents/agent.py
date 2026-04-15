from collections.abc import Callable
from typing import Tuple
import random

from game.enums import loc_after_direction
from game import board, move, enums
from .rat_belief import RatBelief


class PlayerAgent:

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.rat = RatBelief(transition_matrix)
        self.tt = {}
        self.turn = 0

    def commentate(self):
        return "DUBIOUS Angent"

    # ---------------------------
    def play(self, board: board.Board, sensor_data: Tuple, time_left: Callable):

        noise, dist = sensor_data

        self.rat.predict()
        self.rat.update(noise, dist, board)

        best_loc, best_prob = self.rat.get_most_likely()

        #Smarter search trigger
        if best_prob > 0.75:
            return move.Move.search(best_loc)

        moves = board.get_valid_moves()
        if not moves:
            return None

        best_move = random.choice(moves)

        depth = 1

        while depth <= self.dynamic_depth(board) and time_left() > 0.05:

            best_score = float("-inf")

            ordered = self.order_moves(board, moves)

            #Adaptive beam width
            beam_width = 6 if depth <= 2 else 4
            ordered = ordered[:beam_width]

            for m in ordered:
                new_board = board.forecast_move(m)
                if new_board is None:
                    continue

                new_board.reverse_perspective()

                score = self.expectimax(new_board, depth - 1, False)

                if score > best_score:
                    best_score = score
                    best_move = m

            depth += 1

        return best_move

    # ---------------------------
    def expectimax(self, board, depth, maximizing):

        if depth == 0 or board.is_game_over():
            return self.evaluate(board)

        key = (self.hash_board(board), depth, maximizing)
        if key in self.tt:
            return self.tt[key]

        moves = board.get_valid_moves(exclude_search=False)
        moves = self.order_moves(board, moves)

        #prune moves
        moves = moves[:5]

        if maximizing:
            value = float("-inf")

            for m in moves:
                new_board = board.forecast_move(m)
                if new_board is None:
                    continue

                new_board.reverse_perspective()

                value = max(value, self.expectimax(new_board, depth - 1, False))

            self.tt[key] = value
            return value

        else:
            total = 0
            count = 0

            for m in moves:
                new_board = board.forecast_move(m)
                if new_board is None:
                    continue

                new_board.reverse_perspective()

                total += self.expectimax(new_board, depth - 1, True)
                count += 1

            value = total / max(1, count)
            self.tt[key] = value
            return value

    # ---------------------------
    def dynamic_depth(self, board):
        filled = bin(board._primed_mask | board._carpet_mask).count("1")

        if filled < 15:
            return 2
        elif filled < 40:
            return 3
        else:
            return 4  #deeper endgame

    # ---------------------------
    def hash_board(self, board):
        return (
            board._primed_mask,
            board._carpet_mask,
            board.player_worker.get_location(),
            board.opponent_worker.get_location(),
        )

    # ---------------------------
    def order_moves(self, board, moves):

        def score(m):
            s = 0

            if m.move_type == enums.MoveType.CARPET:
                s += 300 + (m.roll_length or 0) * 20
            elif m.move_type == enums.MoveType.PRIME:
                s += 120
            elif m.move_type == enums.MoveType.PLAIN:
                s += 40

            return s

        return sorted(moves, key=score, reverse=True)

    # ---------------------------
    def evaluate(self, board: board.Board):

        my_score = board.player_worker.get_points()
        opp_score = board.opponent_worker.get_points()

        score_diff = my_score - opp_score

        prime_count = bin(board._primed_mask).count("1")
        carpet_score = self.carpet_potential(board)

        x, y = board.player_worker.get_location()
        center_dist = abs(x - 3.5) + abs(y - 3.5)

        mobility = len(board.get_valid_moves()) - len(board.get_valid_moves(enemy=True))

        #improved rat scoring
        rat_value = self.rat_expected_value(board)

        #endgame aggression
        endgame_bonus = 2 * score_diff if (40 - self.turn) < 10 else 0

        return (
            7 * score_diff +
            2 * prime_count +
            6 * carpet_score -
            1.2 * center_dist +
            0.6 * mobility +
            5 * rat_value +
            endgame_bonus
        )

    # ---------------------------
    def rat_expected_value(self, board):

        total = 0
        worker = board.player_worker.get_location()

        belief = self.rat.belief  # 1D array (length 64)

        for idx in range(64):
            prob = belief[idx]

            x = idx % 8
            y = idx // 8

            dist = abs(worker[0] - x) + abs(worker[1] - y)

            total += prob * max(0, 6 - dist)

        return total

    # ---------------------------
    def carpet_potential(self, board):
        worker = board.player_worker.get_location()
        best = 0

        for direction in enums.Direction:
            length = 0
            loc = worker

            for _ in range(7):
                loc = loc_after_direction(loc, direction)

                if not board.is_valid_cell(loc):
                    break
                if loc == board.opponent_worker.get_location():
                    break
                if loc == board.player_worker.get_location():
                    break
                if not board.is_cell_carpetable(loc):
                    break

                length += 1

            best = max(best, length)

        return best