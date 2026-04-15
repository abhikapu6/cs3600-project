"""Run N parallel games of Albrecht vs Yolanda; report crashes/timeouts/winrate.

Uses subprocess-based parallelism (not multiprocessing) to avoid macOS fork/daemon
issues with JAX and the engine's own process management.
"""

import json
import os
import pathlib
import subprocess
import sys
import time
import traceback


THIS_FILE = pathlib.Path(__file__).resolve()
TOP = THIS_FILE.parents[3]


def run_single(seed: int, swap: bool, limit_resources: bool) -> dict:
    """Play one game in-process. Called when invoked with --single."""
    sys.path.insert(0, str(TOP / "engine"))
    sys.path.insert(0, str(TOP / "3600-agents"))
    import random
    random.seed(seed)
    from gameplay import play_game

    play_dir = str(TOP / "3600-agents")
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


def launch_worker(seed: int, swap: bool, limit: bool):
    args = [sys.executable, str(THIS_FILE), "--single", str(seed),
            "1" if swap else "0", "1" if limit else "0"]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--single":
        seed = int(sys.argv[2])
        swap = sys.argv[3] == "1"
        limit = sys.argv[4] == "1"
        r = run_single(seed, swap, limit)
        print(json.dumps(r))
        return

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    limit = "--limit" in sys.argv

    out_path = THIS_FILE.parent / "stress_results.jsonl"
    print(f"Running {n} games, {workers} workers, limit_resources={limit}", flush=True)
    t0 = time.perf_counter()
    crashes = wins = losses = ties = 0

    # Task queue
    tasks = [(i, i % 2 == 1, limit) for i in range(n)]
    active = {}  # proc -> (idx, seed, swap)
    next_task = 0
    completed = 0

    with open(out_path, "w") as f:
        while next_task < n or active:
            # Fill up workers
            while len(active) < workers and next_task < n:
                seed, swap, lim = tasks[next_task]
                proc = launch_worker(seed, swap, lim)
                active[proc] = (next_task, seed, swap)
                next_task += 1

            # Wait for any to finish
            done_procs = []
            for proc in list(active.keys()):
                if proc.poll() is not None:
                    done_procs.append(proc)
            if not done_procs:
                time.sleep(0.5)
                continue

            for proc in done_procs:
                idx, seed, swap = active.pop(proc)
                completed += 1
                out, err = proc.communicate()
                try:
                    r = json.loads(out.decode().strip().splitlines()[-1])
                except Exception:
                    r = {"seed": seed, "ok": False, "swap": swap,
                         "err": f"bad stdout: {out!r} / stderr: {err!r}"}
                f.write(json.dumps(r) + "\n")
                f.flush()
                if not r.get("ok"):
                    crashes += 1
                    err_line = r.get("err", "?").splitlines()[-1] if r.get("err") else "?"
                    print(f"[{completed}/{n}] CRASH seed={seed}: {err_line}", flush=True)
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
                print(f"[{completed}/{n}] seed={r['seed']} swap={r['swap']} "
                      f"winner={winner} pts={r['a_pts']}-{r['b_pts']} "
                      f"turns={r['turns']} wall={r['wall']:.1f}s", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n=== {n} games in {elapsed:.1f}s ===", flush=True)
    print(f"Albrecht: {wins}W {losses}L {ties}T  crashes={crashes}", flush=True)
    print(f"results written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
