"""Heuristic evaluation function for board states (v2).

Evaluates from the perspective of board.player_worker (the side to move).
Features: score_delta, carpet_potential, future_carpet_potential,
          mobility, setup_distance, dead_prime_penalty.
"""

from game.enums import BOARD_SIZE, CARPET_POINTS_TABLE
from .weights import (
    SCORE_DELTA_W, CARPET_POTENTIAL_W, FUTURE_CARPET_W,
    MOBILITY_W, SETUP_DISTANCE_W, DEAD_PRIME_PENALTY_W,
)

# Direction offsets: (dx, dy) for UP, RIGHT, DOWN, LEFT
_DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

# Precompute bit indices for all cells
_ALL_BITS = [(x, y, 1 << (y * BOARD_SIZE + x))
             for y in range(BOARD_SIZE) for x in range(BOARD_SIZE)]


def _worker_mask(board):
    """Return bitmask of both worker positions."""
    p = board.player_worker.get_location()
    o = board.opponent_worker.get_location()
    m = 0
    if p[0] >= 0:
        m |= 1 << (p[1] * BOARD_SIZE + p[0])
    if o[0] >= 0:
        m |= 1 << (o[1] * BOARD_SIZE + o[0])
    return m


def carpet_potential(board, worker_loc):
    """Best carpet score achievable from worker_loc by rolling over primed cells."""
    best = 0
    x, y = worker_loc
    primed = board._primed_mask
    wm = _worker_mask(board)

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


def future_carpet_potential(board, worker_loc):
    """Carpet potential assuming up to 2 future prime steps in a straight line.

    For each direction, extend the ray past existing primed cells by up to 2
    hypothetical primes on SPACE cells, discounting by gamma^k. Returns best value.
    """
    gamma = 0.6
    best = 0.0
    x, y = worker_loc
    primed = board._primed_mask
    blocked = board._blocked_mask
    carpet = board._carpet_mask
    wm = _worker_mask(board)

    for dx, dy in _DIRS:
        cx, cy = x, y
        count = 0
        future_steps = 0
        discount = 1.0
        for _ in range(BOARD_SIZE - 1):
            cx += dx
            cy += dy
            if cx < 0 or cx >= BOARD_SIZE or cy < 0 or cy >= BOARD_SIZE:
                break
            bit = 1 << (cy * BOARD_SIZE + cx)
            if wm & bit:
                break
            if primed & bit:
                count += 1
            elif future_steps < 2 and not ((blocked | carpet) & bit):
                # Hypothetical future prime
                count += 1
                future_steps += 1
                discount *= gamma
            else:
                break

        if count > 0 and count in CARPET_POINTS_TABLE:
            val = CARPET_POINTS_TABLE[count] * discount
            if val > best:
                best = val
    return best


def mobility(board, worker_loc):
    """Count reachable cells within manhattan distance 2 via plain steps.

    BFS on cells not blocked to movement (space or carpet, not primed/blocked/occupied).
    """
    x0, y0 = worker_loc
    blocked = board._blocked_mask | board._primed_mask
    wm = _worker_mask(board)
    occupied = blocked | wm

    visited = set()
    visited.add((x0, y0))
    frontier = [(x0, y0)]
    count = 0

    for depth in range(2):
        next_frontier = []
        for fx, fy in frontier:
            for dx, dy in _DIRS:
                nx, ny = fx + dx, fy + dy
                if nx < 0 or nx >= BOARD_SIZE or ny < 0 or ny >= BOARD_SIZE:
                    continue
                if (nx, ny) in visited:
                    continue
                bit = 1 << (ny * BOARD_SIZE + nx)
                if occupied & bit:
                    continue
                visited.add((nx, ny))
                next_frontier.append((nx, ny))
                count += 1
        frontier = next_frontier

    return count


def setup_distance(board, worker_loc):
    """Manhattan distance to the best latent carpet run start.

    Finds the primed cell that is the start of the longest primed ray
    and returns the manhattan distance from the worker to it.
    Returns 0 if no primed cells exist.
    """
    primed = board._primed_mask
    if primed == 0:
        return 0

    wm = _worker_mask(board)
    best_run = 0
    best_start = None

    # For each primed cell, check 4 ray directions for consecutive primed
    for x, y, bit in _ALL_BITS:
        if not (primed & bit):
            continue
        for dx, dy in _DIRS:
            cx, cy = x, y
            count = 0
            for _ in range(BOARD_SIZE - 1):
                cx += dx
                cy += dy
                if cx < 0 or cx >= BOARD_SIZE or cy < 0 or cy >= BOARD_SIZE:
                    break
                cbit = 1 << (cy * BOARD_SIZE + cx)
                if (primed & cbit) and not (wm & cbit):
                    count += 1
                else:
                    break
            if count > best_run:
                best_run = count
                # The "start" is the cell adjacent to this primed run
                # (where worker needs to be to carpet)
                # That's (x - dx, y - dy) from the first primed cell
                sx, sy = x - dx, y - dy
                if 0 <= sx < BOARD_SIZE and 0 <= sy < BOARD_SIZE:
                    best_start = (sx, sy)

    if best_start is None:
        return 0

    wx, wy = worker_loc
    return abs(wx - best_start[0]) + abs(wy - best_start[1])


def dead_prime_penalty(board):
    """Count primed cells that have no viable extension in any direction.

    A primed cell is "dead" if in every direction, the adjacent cell is either
    out of bounds, blocked, carpet, or another worker — meaning it can never
    become part of a carpet roll from any approach.
    """
    primed = board._primed_mask
    if primed == 0:
        return 0

    blocked = board._blocked_mask
    carpet = board._carpet_mask
    wm = _worker_mask(board)
    obstacle = blocked | carpet | wm

    count = 0
    for x, y, bit in _ALL_BITS:
        if not (primed & bit):
            continue
        dead = True
        for dx, dy in _DIRS:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= BOARD_SIZE or ny < 0 or ny >= BOARD_SIZE:
                continue
            nbit = 1 << (ny * BOARD_SIZE + nx)
            # If neighbor is space or primed (not obstacle), there's a viable direction
            if not (obstacle & nbit):
                dead = False
                break
            # If neighbor is also primed, the chain can extend
            if primed & nbit:
                dead = False
                break
        if dead:
            count += 1
    return count


def evaluate(board):
    """Return heuristic score from player_worker's perspective."""
    my_points = board.player_worker.get_points()
    opp_points = board.opponent_worker.get_points()

    my_loc = board.player_worker.get_location()
    opp_loc = board.opponent_worker.get_location()

    my_cp = carpet_potential(board, my_loc)
    opp_cp = carpet_potential(board, opp_loc)

    my_fcp = future_carpet_potential(board, my_loc)
    opp_fcp = future_carpet_potential(board, opp_loc)

    my_mob = mobility(board, my_loc)
    opp_mob = mobility(board, opp_loc)

    my_sd = setup_distance(board, my_loc)
    opp_sd = setup_distance(board, opp_loc)

    dpp = dead_prime_penalty(board)

    score = (SCORE_DELTA_W * (my_points - opp_points)
             + CARPET_POTENTIAL_W * (my_cp - opp_cp)
             + FUTURE_CARPET_W * (my_fcp - opp_fcp)
             + MOBILITY_W * (my_mob - opp_mob)
             + SETUP_DISTANCE_W * (my_sd - opp_sd)
             + DEAD_PRIME_PENALTY_W * dpp)

    return score
