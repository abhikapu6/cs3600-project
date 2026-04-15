"""Run N parallel games of Albrecht vs Yolanda; report crashes/timeouts/winrate."""

import json
import os
import pathlib
import sys
import time
import traceback
from multiprocessing import Pool


def run_one(args):
    seed, swap, limit_resources = args
    top = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(top / "engine"))
    sys.path.insert(0, str(top / "3600-agents"))
    import random
    random.seed(seed)
    try:
        from gameplay import play_game
    except Exception:
        return {"seed": seed, "ok": False, "err": traceback.format_exc()}
    play_dir = str(top / "3600-agents")
    a, b = ("Albrecht", "Yolanda") if not swap else ("Yolanda", "Albrecht")
    t0 = time.perf_counter()
    try:
        result = play_game(
            play_dir, play_dir, a, b,
            display_game=False, delay=0.0, clear_screen=False,
            record=False, limit_resources=limit_resources,
        )
        board = result[0]
        a_pts = board.player_a.get_points() if hasattr(board, "player_a") else None
        b_pts = board.player_b.get_points() if hasattr(board, "player_b") else None
        winner_attr = getattr(board, "winner", None)
        return {
            "seed": seed, "ok": True, "swap": swap,
            "winner": str(winner_attr),
            "a_pts": a_pts, "b_pts": b_pts,
            "turns": getattr(board, "turn_count", None),
            "wall": time.perf_counter() - t0,
        }
    except Exception:
        return {"seed": seed, "ok": False, "swap": swap,
                "err": traceback.format_exc(),
                "wall": time.perf_counter() - t0}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    limit = "--limit" in sys.argv
    args = [(i, i % 2 == 1, limit) for i in range(n)]
    out_path = pathlib.Path(__file__).parent / "stress_results.jsonl"
    print(f"Running {n} games, {workers} workers, limit_resources={limit}")
    t0 = time.perf_counter()
    crashes = 0
    wins = 0
    losses = 0
    ties = 0
    with Pool(workers) as p, open(out_path, "w") as f:
        for i, r in enumerate(p.imap_unordered(run_one, args)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if not r.get("ok"):
                crashes += 1
                print(f"[{i+1}/{n}] CRASH seed={r['seed']}: {r.get('err','?').splitlines()[-1]}")
                continue
            albrecht_is_a = not r["swap"]
            winner = r["winner"]
            if "PLAYER_A" in winner:
                wins += albrecht_is_a
                losses += not albrecht_is_a
            elif "PLAYER_B" in winner:
                wins += not albrecht_is_a
                losses += albrecht_is_a
            else:
                ties += 1
            print(f"[{i+1}/{n}] seed={r['seed']} swap={r['swap']} "
                  f"winner={winner} pts={r['a_pts']}-{r['b_pts']} "
                  f"turns={r['turns']} wall={r['wall']:.1f}s")
    elapsed = time.perf_counter() - t0
    print(f"\n=== {n} games in {elapsed:.1f}s ===")
    print(f"Albrecht: {wins}W {losses}L {ties}T  crashes={crashes}")
    print(f"results written to {out_path}")


if __name__ == "__main__":
    main()
