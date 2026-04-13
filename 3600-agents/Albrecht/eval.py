"""Heuristic evaluation function for board states (v1).

Evaluates from the perspective of board.player_worker (the side to move).
"""

from game.enums import BOARD_SIZE, CARPET_POINTS_TABLE
from .weights import SCORE_DELTA_W, CARPET_POTENTIAL_W


# Direction offsets: (dx, dy) for UP, RIGHT, DOWN, LEFT
_DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def carpet_potential(board, worker_loc):
    """Best carpet score achievable from worker_loc by rolling over primed cells.

    Walks each of 4 rays from worker_loc, counts consecutive primed cells
    (not occupied by either worker), and returns the best CARPET_POINTS_TABLE value.
    """
    best = 0
    x, y = worker_loc
    primed = board._primed_mask

    p_loc = board.player_worker.get_location()
    o_loc = board.opponent_worker.get_location()
    # Precompute worker bits to exclude
    worker_bits = 0
    if p_loc[0] >= 0:
        worker_bits |= 1 << (p_loc[1] * BOARD_SIZE + p_loc[0])
    if o_loc[0] >= 0:
        worker_bits |= 1 << (o_loc[1] * BOARD_SIZE + o_loc[0])

    for dx, dy in _DIRS:
        cx, cy = x, y
        count = 0
        for _ in range(BOARD_SIZE - 1):
            cx += dx
            cy += dy
            if cx < 0 or cx >= BOARD_SIZE or cy < 0 or cy >= BOARD_SIZE:
                break
            bit = 1 << (cy * BOARD_SIZE + cx)
            if (primed & bit) and not (worker_bits & bit):
                count += 1
            else:
                break

        if count > 0 and count in CARPET_POINTS_TABLE:
            val = CARPET_POINTS_TABLE[count]
            if val > best:
                best = val

    return best


def evaluate(board):
    """Return heuristic score from player_worker's perspective.

    Positive = good for player_worker, negative = good for opponent_worker.
    """
    my_points = board.player_worker.get_points()
    opp_points = board.opponent_worker.get_points()

    my_loc = board.player_worker.get_location()
    opp_loc = board.opponent_worker.get_location()

    my_cp = carpet_potential(board, my_loc)
    opp_cp = carpet_potential(board, opp_loc)

    score = (SCORE_DELTA_W * (my_points - opp_points)
             + CARPET_POTENTIAL_W * (my_cp - opp_cp))

    return score
