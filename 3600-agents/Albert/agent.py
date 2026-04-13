"""Albert stub — expectiminimax depth 3 + simple eval.

Used as a D3 milestone comparator. NOT shipped in the submission.
To use: symlink or copy as 3600-agents/Albert/agent.py.
"""

import time
from collections.abc import Callable
from typing import Tuple

from game import board, move, enums
from game.enums import MoveType, CARPET_POINTS_TABLE, Direction, BOARD_SIZE


_DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def _carpet_potential(board, worker_loc):
    best = 0
    x, y = worker_loc
    primed = board._primed_mask
    p_loc = board.player_worker.get_location()
    o_loc = board.opponent_worker.get_location()
    wm = 0
    if p_loc[0] >= 0:
        wm |= 1 << (p_loc[1] * BOARD_SIZE + p_loc[0])
    if o_loc[0] >= 0:
        wm |= 1 << (o_loc[1] * BOARD_SIZE + o_loc[0])

    for dx, dy in _DIRS:
        cx, cy = x, y
        count = 0
        for _ in range(BOARD_SIZE - 1):
            cx += dx
            cy += dy
            if cx < 0 or cx >= BOARD_SIZE or cy < 0 or cy >= BOARD_SIZE:
                break
            bit = 1 << (cy * BOARD_SIZE + cx)
            if (primed & bit) and not (wm & bit):
                count += 1
            else:
                break
        if count > 0 and count in CARPET_POINTS_TABLE:
            val = CARPET_POINTS_TABLE[count]
            if val > best:
                best = val
    return best


def _evaluate(board):
    my_pts = board.player_worker.get_points()
    opp_pts = board.opponent_worker.get_points()
    my_loc = board.player_worker.get_location()
    opp_loc = board.opponent_worker.get_location()
    my_cp = _carpet_potential(board, my_loc)
    opp_cp = _carpet_potential(board, opp_loc)
    return 1.0 * (my_pts - opp_pts) + 0.8 * (my_cp - opp_cp)


def _order_moves(moves):
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


class _Timeout(Exception):
    pass


def _negamax(board, depth, alpha, beta, deadline):
    if time.monotonic() >= deadline:
        raise _Timeout

    if board.is_game_over():
        my_pts = board.player_worker.get_points()
        opp_pts = board.opponent_worker.get_points()
        diff = my_pts - opp_pts
        if diff > 0:
            return 1000.0 + diff
        elif diff < 0:
            return -1000.0 + diff
        return 0.0

    if depth <= 0:
        return _evaluate(board)

    moves = board.get_valid_moves(exclude_search=True)
    if not moves:
        return _evaluate(board)

    moves = _order_moves(moves)
    best = float('-inf')

    for m in moves:
        child = board.forecast_move(m)
        if child is None:
            continue
        child.reverse_perspective()
        score = -_negamax(child, depth - 1, -beta, -alpha, deadline)
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best if best > float('-inf') else _evaluate(board)


class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.turn = 0

    def commentate(self):
        return "I am Albert"

    def play(self, board: board.Board, sensor_data: Tuple, time_left: Callable):
        self.turn += 1
        remaining = time_left()
        turns_left = max(1, board.player_worker.turns_left)
        budget = max(0.5, (remaining - 15) / turns_left)
        deadline = time.monotonic() + 0.9 * budget

        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            return move.Move.plain(Direction.UP)

        moves = _order_moves(moves)
        best_move = moves[0]
        best_score = float('-inf')

        # Fixed depth 3
        target_depth = 3
        try:
            for m in moves:
                if time.monotonic() >= deadline:
                    break
                child = board.forecast_move(m)
                if child is None:
                    continue
                child.reverse_perspective()
                score = -_negamax(child, target_depth - 1, float('-inf'), float('inf'), deadline)
                if score > best_score:
                    best_score = score
                    best_move = m
        except _Timeout:
            pass

        return best_move
