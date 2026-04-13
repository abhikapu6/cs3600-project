"""Quick batch test: run N games between two agents."""
import os, sys, pathlib, multiprocessing

def main():
    engine_dir = pathlib.Path(__file__).parent / "engine"
    sys.path.insert(0, str(engine_dir))
    os.chdir(str(engine_dir))

    from gameplay import play_game

    top_level = pathlib.Path(__file__).parent.resolve()
    play_dir = str(top_level / "3600-agents")
    a_name, b_name = sys.argv[1], sys.argv[2]
    n_games = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    wins_a, wins_b, ties = 0, 0, 0
    for i in range(n_games):
        try:
            fb, *_ = play_game(play_dir, play_dir, a_name, b_name,
                               display_game=False, delay=0, clear_screen=False,
                               record=False, limit_resources=False)
            w = fb.get_winner()
            pa = fb.player_worker.get_points()
            pb = fb.opponent_worker.get_points()
            if w.value == 0:  # PLAYER_A
                wins_a += 1
                tag = "A"
            elif w.value == 1:  # PLAYER_B
                wins_b += 1
                tag = "B"
            else:
                ties += 1
                tag = "T"
            print(f"Game {i+1}: {tag}  (A={pa} B={pb})", flush=True)
        except Exception as e:
            print(f"Game {i+1}: ERROR {e}", flush=True)

    print(f"\nResults: {a_name} {wins_a}W / {b_name} {wins_b}W / {ties}T  ({n_games} games)")
    print(f"{a_name} win rate: {wins_a/n_games*100:.0f}%")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
