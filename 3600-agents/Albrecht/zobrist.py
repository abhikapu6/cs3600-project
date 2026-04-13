"""Zobrist hashing for transposition table keys."""

import random

from game.enums import BOARD_SIZE

N = BOARD_SIZE * BOARD_SIZE  # 64

# Deterministic seed for reproducibility
_rng = random.Random(0xALBRECHT)

# Keys for cell types at each position: [4 types][64 positions]
# Types: 0=SPACE, 1=PRIMED, 2=CARPET, 3=BLOCKED
CELL_KEYS = [[_rng.getrandbits(64) for _ in range(N)] for _ in range(4)]

# Keys for worker positions
PLAYER_LOC_KEYS = [_rng.getrandbits(64) for _ in range(N)]
OPP_LOC_KEYS = [_rng.getrandbits(64) for _ in range(N)]

# Side to move
SIDE_KEY = _rng.getrandbits(64)


def board_hash(board):
    """Compute Zobrist hash for a board state."""
    h = 0

    for i in range(N):
        bit = 1 << i
        if board._primed_mask & bit:
            h ^= CELL_KEYS[1][i]
        elif board._carpet_mask & bit:
            h ^= CELL_KEYS[2][i]
        elif board._blocked_mask & bit:
            h ^= CELL_KEYS[3][i]
        # SPACE → skip (XOR with 0 is identity)

    p_loc = board.player_worker.get_location()
    o_loc = board.opponent_worker.get_location()
    h ^= PLAYER_LOC_KEYS[p_loc[1] * BOARD_SIZE + p_loc[0]]
    h ^= OPP_LOC_KEYS[o_loc[1] * BOARD_SIZE + o_loc[0]]

    if not board.is_player_a_turn:
        h ^= SIDE_KEY

    return h
