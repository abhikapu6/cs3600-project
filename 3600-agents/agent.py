from collections.abc import Callable
from typing import Tuple
import time

from game.enums import loc_after_direction
from game import board, move, enums
from .rat_belief import RatBelief

class TimeOutException(Exception):
    pass

class PlayerAgent:

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.rat = RatBelief(transition_matrix)
        self.turn = 0
        self.pv_move = None
        self.killer_moves = {}
        self.tt = {} 

    def commentate(self):
        return "Dubious13"

    def play(self, board: board.Board, sensor_data: Tuple, time_left: Callable):
        self.turn += 1
        start_time = time.time()

        noise, dist = sensor_data

        self.rat.predict()
        self.rat.update(noise, dist, board)
        best_rat_loc, best_rat_prob = self.rat.get_most_likely()

        remaining = time_left()
        turns_left = max(1, board.player_worker.turns_left)
        base_budget = remaining / turns_left
        
        if 5 <= self.turn <= 35:
            budget = min(5.0, base_budget * 1.5)
        else:
            budget = min(2.0, base_budget * 0.9)

        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            return None

        ordered_moves = self.order_moves(board, moves, best_rat_loc)
        if self.pv_move in ordered_moves:
            ordered_moves.remove(self.pv_move)
            ordered_moves.insert(0, self.pv_move)

        best_move = ordered_moves[0]
        depth = 1

        try:
            while depth <= 25:
                alpha = float("-inf")
                beta = float("inf")
                current_best_move = None
                current_best_score = float("-inf")

                for m in ordered_moves:
                    if (time.time() - start_time) > budget:
                        raise TimeOutException()

                    new_board = board.forecast_move(m)
                    if new_board is None:
                        continue

                    new_board.reverse_perspective()
                    score = -self.alphabeta(new_board, depth - 1, -beta, -alpha, start_time, budget, best_rat_loc)

                    if score > current_best_score:
                        current_best_score = score
                        current_best_move = m

                    alpha = max(alpha, score)

                # Only update best_move if the entire depth completed without timing out
                best_move = current_best_move
                self.pv_move = best_move
                depth += 1

        except TimeOutException:
            pass 

        should_search = False
        my_loc = board.player_worker.get_location()
        dist_to_rat = abs(my_loc[0] - best_rat_loc[0]) + abs(my_loc[1] - best_rat_loc[1])
        opp_loc = board.opponent_worker.get_location()
        opp_dist_to_rat = abs(opp_loc[0] - best_rat_loc[0]) + abs(opp_loc[1] - best_rat_loc[1])

        # Smart Searching: Only search if we are winning the race to the rat and it is close
        if dist_to_rat <= opp_dist_to_rat:
            if best_move.move_type == enums.MoveType.PLAIN and best_rat_prob > 0.50 and dist_to_rat <= 3:  
                should_search = True
            elif best_move.move_type == enums.MoveType.PRIME and best_rat_prob > 0.70 and dist_to_rat <= 2:  
                should_search = True
            elif best_move.move_type == enums.MoveType.CARPET and getattr(best_move, "roll_length", 0) <= 2 and best_rat_prob > 0.85:
                should_search = True

        if turns_left <= 2 and getattr(best_move, "roll_length", 0) >= 2:
            should_search = False

        if should_search:
            return move.Move.search(best_rat_loc)

        return best_move

    def _hash_state(self, board):
        my_score = board.player_worker.get_points()
        opp_score = board.opponent_worker.get_points()
        return (
            board.player_worker.get_location(),
            board.opponent_worker.get_location(),
            board._primed_mask,
            board._carpet_mask,
            my_score - opp_score
        )

    def alphabeta(self, board, depth, alpha, beta, start_time, budget, target_loc):
        if (time.time() - start_time) > budget:
            raise TimeOutException()

        state_hash = self._hash_state(board)
        tt_entry = self.tt.get(state_hash)
        tt_move = None
        
        if tt_entry is not None:
            tt_move = tt_entry.get('best_move') # Retrieve the cached best move
            if tt_entry['depth'] >= depth:
                if tt_entry['flag'] == 'EXACT':
                    return tt_entry['value']
                elif tt_entry['flag'] == 'LOWERBOUND':
                    alpha = max(alpha, tt_entry['value'])
                elif tt_entry['flag'] == 'UPPERBOUND':
                    beta = min(beta, tt_entry['value'])
                
                if alpha >= beta:
                    return tt_entry['value']

        if board.is_game_over() or depth == 0:
            return self.evaluate(board)

        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            return self.evaluate(board)

        ordered = self.order_moves(board, moves, target_loc)
        
        if tt_move in ordered:
            ordered.remove(tt_move)
            ordered.insert(0, tt_move)
        else:
            killer = self.killer_moves.get(depth)
            if killer in ordered:
                ordered.remove(killer)
                ordered.insert(0, killer)

        ordered = ordered[:10] 
        best = float("-inf")
        original_alpha = alpha 
        current_best_move = None

        for m in ordered:
            new_board = board.forecast_move(m)
            if new_board is None:
                continue

            new_board.reverse_perspective()
            val = -self.alphabeta(new_board, depth - 1, -beta, -alpha, start_time, budget, target_loc)
            
            if val > best:
                best = val
                current_best_move = m

            alpha = max(alpha, val)
            
            if alpha >= beta:
                self.killer_moves[depth] = m
                break

        flag = 'EXACT'
        if best <= original_alpha:
            flag = 'UPPERBOUND'
        elif best >= beta:
            flag = 'LOWERBOUND'

        # Store the best move so we can use it for ordering next time
        self.tt[state_hash] = {'value': best, 'depth': depth, 'flag': flag, 'best_move': current_best_move}
        return best

    def evaluate(self, board: board.Board):
        my_score = board.player_worker.get_points()
        opp_score = board.opponent_worker.get_points()
        score_diff = my_score - opp_score

        my_loc = board.player_worker.get_location()
        opp_loc = board.opponent_worker.get_location()

        my_pts, my_run, my_dir = self._advanced_geometry(board, my_loc)
        opp_pts, opp_run, opp_dir = self._advanced_geometry(board, opp_loc)

        my_territory = self._get_territory(board, my_loc)
        opp_territory = self._get_territory(board, opp_loc)

        top_cells = self.rat.get_top_k(2)
        my_rat_score = 0
        opp_rat_score = 0
        
        best_rat = top_cells[0][0] if top_cells else my_loc
        my_dist_to_rat = abs(my_loc[0] - best_rat[0]) + abs(my_loc[1] - best_rat[1])
        opp_dist_to_rat = abs(opp_loc[0] - best_rat[0]) + abs(opp_loc[1] - best_rat[1])

        # Rat Race Logic
        if my_dist_to_rat <= opp_dist_to_rat and my_dist_to_rat <= 4:
            for (rx, ry), prob in top_cells:
                my_dist = abs(my_loc[0] - rx) + abs(my_loc[1] - ry)
                my_rat_score += prob * (15 - my_dist)
        elif opp_dist_to_rat < my_dist_to_rat:
            for (rx, ry), prob in top_cells:
                opp_dist = abs(opp_loc[0] - rx) + abs(opp_loc[1] - ry)
                opp_rat_score += prob * (15 - opp_dist)

        turns_left = max(1, board.player_worker.turns_left)
        if turns_left <= 4:
            # Physically impossible to build and roll a new line. Focus 100% on cashing out.
            return (
                10000 * score_diff +   
                0 * (my_pts - opp_pts) + # Potential points are an illusion now
                10 * (my_run - opp_run) +               
                10 * (my_territory - opp_territory) +   
                10 * (my_rat_score - opp_rat_score)
            )

        return (
            1200 * score_diff +                      
            800 * (my_pts - opp_pts) +               
            100 * (my_run - opp_run) +               
            100 * (my_territory - opp_territory) +   
            10 * (my_rat_score - opp_rat_score)      
        )

    def _opposite_dir(self, d):
        if d == enums.Direction.UP: return enums.Direction.DOWN
        if d == enums.Direction.DOWN: return enums.Direction.UP
        if d == enums.Direction.LEFT: return enums.Direction.RIGHT
        if d == enums.Direction.RIGHT: return enums.Direction.LEFT
        return None

    def order_moves(self, board, moves, target_loc):
        turns_left = board.player_worker.turns_left
        opp_loc = board.opponent_worker.get_location()
        my_loc = board.player_worker.get_location() 
        
        my_rat_dist = abs(my_loc[0] - target_loc[0]) + abs(my_loc[1] - target_loc[1])
        opp_rat_dist = abs(opp_loc[0] - target_loc[0]) + abs(opp_loc[1] - target_loc[1])
        winning_rat_race = (my_rat_dist <= opp_rat_dist) and (my_rat_dist <= 4)

        def score(m):
            if m.move_type == enums.MoveType.CARPET:
                if m.roll_length <= 1:
                    return -5000 
                
                lengths = {2:2, 3:4, 4:6, 5:10, 6:15, 7:21}
                base = 10000 + lengths.get(m.roll_length, 0) * 1000 
                
                if m.roll_length >= 4:
                    base += 20000

                end_loc = my_loc
                for _ in range(m.roll_length):
                    end_loc = loc_after_direction(end_loc, m.direction)
                    
                dist_to_opp = abs(end_loc[0] - opp_loc[0]) + abs(end_loc[1] - opp_loc[1])
                
                if winning_rat_race:
                    dist_to_target = abs(end_loc[0] - target_loc[0]) + abs(end_loc[1] - target_loc[1])
                    base -= (dist_to_target * 20) 
                
                if m.roll_length >= 3 and dist_to_opp <= 2:
                    base += 5000 
                
                if turns_left <= 2 and m.roll_length >= 2:
                    return base + 50000 
                return base
                
            elif m.move_type == enums.MoveType.PRIME:
                if turns_left <= 1:
                    return -1000 
                
                base_prime = 800
                
                inv_d = self._opposite_dir(m.direction)
                check_loc = loc_after_direction(my_loc, inv_d)
                
                if board.is_valid_cell(check_loc):
                    idx = check_loc[1]*8 + check_loc[0]
                    if (board._primed_mask >> idx) & 1:
                        base_prime += 400 
                
                end_loc = loc_after_direction(my_loc, m.direction)
                
                if winning_rat_race:
                    dist_to_target = abs(end_loc[0] - target_loc[0]) + abs(end_loc[1] - target_loc[1])
                    return base_prime - (dist_to_target * 10) 
                return base_prime
            else:
                return 100

        return sorted(moves, key=score, reverse=True)

    def _get_territory(self, board, loc):
        free_spaces = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = loc[0] + dx, loc[1] + dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    idx = ny * 8 + nx
                    is_blocked = (board._blocked_mask >> idx) & 1
                    is_carpet = (board._carpet_mask >> idx) & 1
                    if not is_blocked and not is_carpet:
                        free_spaces += 1
        return free_spaces

    def _advanced_geometry(self, board, loc):
        best_points = 0
        best_runway = 0
        best_dir = None
        scale = {0:0, 1:-1, 2:2, 3:4, 4:6, 5:10, 6:15, 7:21}

        for d in enums.Direction:
            length = 0
            runway = 0
            curr_loc = loc

            for _ in range(7):
                curr_loc = loc_after_direction(curr_loc, d)
                if not board.is_valid_cell(curr_loc):
                    break

                idx = curr_loc[1]*8 + curr_loc[0]
                if ((board._primed_mask >> idx) & 1):
                    length += 1
                else:
                    temp_loc = curr_loc
                    for _ in range(7 - length): 
                        if not board.is_valid_cell(temp_loc):
                            break
                        
                        t_idx = temp_loc[1]*8 + temp_loc[0]
                        is_blocked = (board._blocked_mask >> t_idx) & 1
                        is_carpet = (board._carpet_mask >> t_idx) & 1
                        
                        if not is_blocked and not is_carpet:
                            runway += 1
                            temp_loc = loc_after_direction(temp_loc, d)
                        else:
                            break 
                    break 

            points = scale.get(length, 0)
            current_eval = points + (runway * 0.5)
            
            if current_eval > (best_points + (best_runway * 0.5)):
                best_points = points
                best_runway = runway
                best_dir = d

        return best_points, best_runway, best_dir