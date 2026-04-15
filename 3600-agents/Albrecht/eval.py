"""Heuristic evaluation function for board states (v3).

Evaluates from the perspective of board.player_worker (the side to move).
v3 features (belief-aware): search_ev_best, belief_entropy,
opponent_disruption, time_pressure, blocked_corner_awareness.
"""

import math

from game.enums import BOARD_SIZE, CARPET_POINTS_TABLE
from .weights import (
    SCORE_DELTA_W, CARPET_POTENTIAL_W, FUTURE_CARPET_W,
    MOBILITY_W, SETUP_DISTANCE_W, DEAD_PRIME_PENALTY_W,
    EXCESS_PRIMES_W, EXCESS_PRIMES_THRESHOLD,
    SEARCH_EV_BEST_W, BELIEF_ENTROPY_W, OPPONENT_DISRUPTION_W,
    TIME_PRESSURE_W, BLOCKED_CORNER_W,
)

_DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

_ALL_BITS = [(x, y, 1 << (y * BOARD_SIZE + x))
             for y in range(BOARD_SIZE) for x in range(BOARD_SIZE)]


def _worker_mask(board):
    p = board.player_worker.get_location()
    o = board.opponent_worker.get_location()
    m = 0
    if p[0] >= 0:
        m |= 1 << (p[1] * BOARD_SIZE + p[0])
    if o[0] >= 0:
        m |= 1 << (o[1] * BOARD_SIZE + o[0])
    return m


def carpet_potential(board, worker_loc):
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
    """Best achievable carpet chain anywhere on the board, discounted by
    future primes needed and Manhattan distance from the worker to the
    chain's entry cell (adjacent to the start)."""
    gamma_prime = 0.85
    gamma_dist = 0.75
    best = 0.0
    primed = board._primed_mask
    blocked = board._blocked_mask
    carpet = board._carpet_mask
    wm = _worker_mask(board)
    obstacle = blocked | carpet | wm
    wx, wy = worker_loc

    for sx, sy, _sbit in _ALL_BITS:
        for dx, dy in _DIRS:
            entry_x, entry_y = sx - dx, sy - dy
            if not (0 <= entry_x < BOARD_SIZE and 0 <= entry_y < BOARD_SIZE):
                continue
            entry_bit = 1 << (entry_y * BOARD_SIZE + entry_x)
            if (blocked | carpet) & entry_bit:
                continue
            cx, cy = sx, sy
            length = 0
            future = 0
            for _ in range(BOARD_SIZE):
                if cx < 0 or cx >= BOARD_SIZE or cy < 0 or cy >= BOARD_SIZE:
                    break
                bit = 1 << (cy * BOARD_SIZE + cx)
                if primed & bit:
                    length += 1
                elif not (obstacle & bit):
                    length += 1
                    future += 1
                else:
                    break
                cx += dx
                cy += dy
            if length < 2 or length not in CARPET_POINTS_TABLE:
                continue
            dist = abs(wx - entry_x) + abs(wy - entry_y)
            val = (CARPET_POINTS_TABLE[length]
                   * (gamma_prime ** future)
                   * (gamma_dist ** dist))
            if val > best:
                best = val
    return best


def mobility(board, worker_loc):
    x0, y0 = worker_loc
    blocked = board._blocked_mask | board._primed_mask
    wm = _worker_mask(board)
    occupied = blocked | wm

    visited = {(x0, y0)}
    frontier = [(x0, y0)]
    count = 0

    for _ in range(2):
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
    primed = board._primed_mask
    if primed == 0:
        return 0

    wm = _worker_mask(board)
    best_run = 0
    best_start = None

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
                sx, sy = x - dx, y - dy
                if 0 <= sx < BOARD_SIZE and 0 <= sy < BOARD_SIZE:
                    best_start = (sx, sy)

    if best_start is None:
        return 0

    wx, wy = worker_loc
    return abs(wx - best_start[0]) + abs(wy - best_start[1])


def dead_prime_penalty(board):
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
            if not (obstacle & nbit):
                dead = False
                break
            if primed & nbit:
                dead = False
                break
        if dead:
            count += 1
    return count


def search_ev_best(belief):
    """Best EV a SEARCH move can produce = 6*max(b) - 2. Clamped at 0."""
    if belief is None:
        return 0.0
    ev = 6.0 * float(belief.b.max()) - 2.0
    return max(ev, 0.0)


def belief_entropy(belief):
    """Shannon entropy (nats) of the belief. Lower = more concentrated."""
    if belief is None:
        return 0.0
    b = belief.b
    total = 0.0
    for p in b:
        if p > 1e-9:
            total -= float(p) * math.log(float(p))
    return total


def opponent_disruption(board, opp_loc):
    """Count of opp's 4 rays that are immediately blocked within 3 cells.

    A high count means opponent's mobility/carpet prospects are constrained.
    """
    blocked = board._blocked_mask | board._carpet_mask
    wm = _worker_mask(board)
    obstacle = blocked | wm
    x, y = opp_loc
    walled = 0
    for dx, dy in _DIRS:
        cx, cy = x, y
        open_run = 0
        for _ in range(3):
            cx += dx
            cy += dy
            if cx < 0 or cx >= BOARD_SIZE or cy < 0 or cy >= BOARD_SIZE:
                break
            bit = 1 << (cy * BOARD_SIZE + cx)
            if obstacle & bit:
                break
            open_run += 1
        if open_run <= 1:
            walled += 1
    return walled


def blocked_corner_awareness(board, worker_loc):
    """Penalty proportional to surrounding blocked mass within radius 2."""
    blocked = board._blocked_mask
    x0, y0 = worker_loc
    count = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dx) + abs(dy) > 2:
                continue
            nx, ny = x0 + dx, y0 + dy
            if nx < 0 or nx >= BOARD_SIZE or ny < 0 or ny >= BOARD_SIZE:
                count += 1
                continue
            bit = 1 << (ny * BOARD_SIZE + nx)
            if blocked & bit:
                count += 1
    return -count


def evaluate(board, belief=None):
    """Return heuristic score from player_worker's perspective.

    belief: optional RatBelief for the side to move (used for v3 features).
    """
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

    total_primes = bin(board._primed_mask).count("1")
    excess_primes = max(0, total_primes - EXCESS_PRIMES_THRESHOLD)

    score = (SCORE_DELTA_W * (my_points - opp_points)
             + CARPET_POTENTIAL_W * (my_cp - opp_cp)
             + FUTURE_CARPET_W * (my_fcp - opp_fcp)
             + MOBILITY_W * (my_mob - opp_mob)
             + SETUP_DISTANCE_W * (my_sd - opp_sd)
             + DEAD_PRIME_PENALTY_W * dpp
             + EXCESS_PRIMES_W * excess_primes)

    # v3 belief-aware features
    score += SEARCH_EV_BEST_W * search_ev_best(belief)
    score += BELIEF_ENTROPY_W * belief_entropy(belief)

    # Disruption: bonus when opp is more walled than us
    my_walled = opponent_disruption(board, my_loc)
    opp_walled = opponent_disruption(board, opp_loc)
    score += OPPONENT_DISRUPTION_W * (opp_walled - my_walled)

    # Time pressure: penalty when we're behind on clock
    my_time = board.player_worker.time_left
    opp_time = board.opponent_worker.time_left
    score += TIME_PRESSURE_W * max(0.0, opp_time - my_time)

    # Blocked-corner awareness
    my_bca = blocked_corner_awareness(board, my_loc)
    opp_bca = blocked_corner_awareness(board, opp_loc)
    score += BLOCKED_CORNER_W * (my_bca - opp_bca)

    return score
