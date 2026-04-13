# Standout Tournament Agent — Plan

## Context

CS3600 Spring 2026 final-project tournament: 8×8 carpet game with hidden rat (HMM). Grade is determined by ELO rank vs. three reference bots — George (70%), Albert (80%, expectiminimax + HMM), Carrie (90%, expectiminimax + HMM + advanced heuristic). The goal is to **beat Carrie decisively AND stand out** from the other teams that will also beat Carrie, ideally contending for the top team prize. Submission deadline is 2026-04-19 (≈9 days from today, 2026-04-10).

The canonical Carrie-beater recipe is already known: expectiminimax + alpha-beta + HMM + handcrafted heuristic. To stand out we must do that *well* and add 2–3 differentiators that most students will miss:

1. **Opponent-search observation exploitation** in the rat belief (their search location + hit/miss is public data most students ignore).
2. **Rat belief integrated inside the search tree**, so `SEARCH` vs carpet vs prime is scored with correct expected value at every node instead of via a bolted-on heuristic.
3. **Self-play-tuned evaluation weights** via a parallel match-runner + coordinate descent / CMA-ES, producing weights the hand-tuned crowd can't match.

Risk profile: prioritize a robust Carrie-beater by D5, then invest remaining time in tuning + endgame exhaustive to push ELO.

## Constraints (verified from codebase)

- Entry point: `3600-agents/<Name>/agent.py` with `PlayerAgent(board, transition_matrix, time_left)`, `play(board, sensor_data, time_left) -> Move`, `commentate()`. Same signature as `3600-agents/Yolanda/agent.py:14`.
- `sensor_data = (Noise, int_distance)`; distance is a noisy Manhattan from our *pre-move* worker position (`engine/game/rat.py:112` + `engine/gameplay.py:387`).
- `transition_matrix` is a `jax.numpy` float32 (64,64) row-stochastic matrix, fixed for the whole game, independent of board cell state (`engine/gameplay.py:10` + `engine/game/rat.py:39`). Convert to numpy at init.
- Rat respawn = fresh (0,0) + 1000 silent moves on same T (`rat.spawn()` in `engine/game/rat.py:127`). So `spawn_dist = T^1000 @ e_{(0,0)}` is a fixed constant per game.
- `board.opponent_search` / `board.player_search` are correctly oriented for us after `reverse_perspective` (`engine/gameplay.py:457`). Value is `((x,y), hit_bool)` or `(None, False)`.
- Opponent's own sensor readings (their noise/distance) are **NOT** shared with us — only their search location + result.
- **Resource sandbox (subprocess)**: 1.5 GB RAM, seccomp (no network, no writes outside cwd), process **paused (SIGSTOP) between turns** — no background thinking. Init budget **10 s** (separate); per-game total budget **240 s** wall for all our moves combined.
- Total move budget ≈ 240s / 40 turns = **~6 s/turn average**; we'll target ~3 s nominal and spend more in critical turns, reserving ~15 s safety buffer.
- **Tournament-machine libraries (guaranteed)**: numpy, PyTorch, JAX, FLAX, Plyvel, scikit-learn, Python 3.12. **Numba is NOT guaranteed** (it's only in the local `requirements.txt`). Shipped agent must stick to numpy / JAX / torch.
- Zip must be ≤200 MB, under the agent directory, with an `__init__.py` alongside `agent.py` for multi-file imports (assignment §7.1).

## Recommended architecture

### Module layout — `3600-agents/Albrecht/`

```
__init__.py        # from .agent import PlayerAgent ; from . import belief, search, ...
agent.py           # thin PlayerAgent: owns belief + searcher, routes play()
belief.py          # 64-state rat HMM forward filter (numpy)
t_precompute.py    # spawn_dist via repeated squaring, sparse T neighbor lists
search.py          # iterative-deepening expectiminimax + alpha-beta + TT + PV
eval.py            # carpet-planning-aware heuristic
zobrist.py         # Zobrist keys + TT entry dataclass
opponent_model.py  # (optional) opp profile from observed moves
endgame.py         # exhaustive deep search for last ~4-6 plies
weights.py         # tuned float constants (shipped)
tools/             # DEV-ONLY, excluded from submission zip
  match_runner.py  # multiprocessing parallel game runner
  tune.py          # coordinate descent / CMA-ES over weights
  regress.py       # smoke tests vs Yolanda/random/mirror
```

### Rat belief — `belief.py`
- State `b: np.float32[64]`. `predict(): b = b @ T_np`.
- `update_sensor(b, noise, dist, worker_pos, board)`: precomputed `manhattan_lut[64,64]`, `dist_lik[observed, true_dist]` (true 0..14, with 0-clamp handled), and per-cell `noise_lik[cell_type, noise]`. Single vectorized `b *= noise_vec * dist_vec ; b /= b.sum()`.
- `update_search(b, loc, hit)`: miss → zero `b[idx(loc)]`, renormalize; hit → `b = spawn_dist.copy()`. Apply to both our past searches (via `board.player_search`) and opponent's (via `board.opponent_search`) — **opponent-search folding is Standout Differentiator #1**.
- `spawn_dist` precomputed in `__init__` via 10 dense matmuls (T^1000 by repeated squaring). Sub-millisecond.
- Belief is cloned (`np.copy`, 256 B) when branching inside the search tree.
- **Order of updates each turn**: predict (rat moves before sample, per `gameplay.py:386-387`) → fold in our sensor reading → fold in opponent's search result from last turn → search/act.

### Search — `search.py`
- Expectiminimax over (us, opp) plies. Our `SEARCH(c)` action is a chance node: `p*(+4 + V(belief_collapsed_or_reset)) + (1-p)*(-2 + V(belief_with_c_zeroed))`, where `p = belief[idx(c)]`. Movement actions are deterministic.
- **Alpha-beta** on max/min layers; chance-node bounds pruning is optional (star1/star2) — defer until profiling.
- **Iterative deepening** 1..D until `time.monotonic() - start >= 0.9 * per_turn_budget`, then return best move from last completed depth.
- **Zobrist TT**: key = `hash(primed_mask, carpet_mask, blocked_mask, player_loc, opp_loc, side_to_move, turn_count)`. Belief is **not** in the key — treat belief-dependent eval as a noisy perturbation, and only trust TT results for move ordering (primary) and lower-depth value bounds (secondary). Size cap 2^18 entries, replace-by-depth.
- **Move ordering**: TT best → carpet (desc by score) → prime (biased toward longest prospective roll) → plain → search (only top-K cells by belief, K=3). Most `SEARCH` moves are pruned aggressively — only expand if `max(belief) > 0.15` or EV > 0.
- **Opponent model** inside search: assume opponent uses a matching evaluator (self-play symmetry).
- **Time budget**: `budget = max(1.0, (remaining_time - 15) / max(1, turns_remaining)) * critical_mult`, where `critical_mult ∈ [0.8, 1.4]` based on turn phase. Last 4–6 turns hand off to `endgame.py`.

### Evaluation — `eval.py`
Handcrafted features, weighted sum, returned as `my - opp` potential so expectiminimax works with a single scalar. All via numpy / bitboard tricks on the engine's existing `_primed_mask`, `_carpet_mask`, `_blocked_mask`:

- `score_delta = my_points - opp_points`
- `carpet_potential(worker)`: for each of 4 rays, walk along primed-mask shifts (mirror of engine's `get_valid_moves` carpet logic) to find max contiguous primed length, score via `CARPET_POINTS_TABLE`. Take the best.
- `future_carpet_potential`: same thing but assuming up to 2 future prime steps in a straight line; discount by `γ^k`.
- `dead_prime_penalty`: primed cells with no viable extension (e.g., cornered by blocked/opp) → negative weight.
- `mobility`: popcount of plain-step-reachable cells within radius 2 (BFS on bitboard).
- `setup_distance`: worker's Manhattan distance to the best latent carpet run start.
- `search_ev_best`: `max_c 6*b[c] - 2` (= best `SEARCH` EV given belief) — makes search attractive when appropriate.
- `belief_entropy`: small negative coefficient — entropy reduction is implicitly rewarded.
- `opponent_disruption`: bonus if our primes intersect opponent's best future carpet ray; penalty if opponent can plain-step onto our half-built chain's starting square.
- `time_pressure`: `−λ * max(0, opp.time_left − self.time_left)`.
- `blocked_corner_awareness`: small bonus for being near space-rich regions (avoid getting cornered by random 2×3 blockers).

All coefficients live in `weights.py`. Engine APIs reused: `board._primed_mask`, `board._carpet_mask`, `board._blocked_mask`, `board._shift_mask_*`, `board.get_valid_moves`, `board.forecast_move` (only when we truly need a new Board — keep inner eval purely on masks).

### Endgame — `endgame.py`
Last 4–6 plies (`min(turns_left*2, 6)`): bypass heuristic, run full alpha-beta to a leaf with exact score delta. Branching is high but with aggressive move ordering + TT seeded from regular search, 4-ply is comfortable; attempt 6-ply with early cutoff on budget.

### Standout differentiators (shipping)

1. **Opponent search folding** (in `belief.py`) — cheap, high-leverage, most students miss it.
2. **Belief-integrated search** (in `search.py`) — `SEARCH` is a first-class chance node with proper EV; no ad-hoc "should I search?" threshold.
3. **Self-play-tuned weights** (dev tools) — see below. Delivers a principled advantage over hand-tuned crowd.

Deliberately dropped:
- **AlphaZero-lite policy/value NN**: not feasible to train, debug, and ship in 9 days alongside the rest.
- **Opening book**: replaced with a 3-turn scripted "carpet setup" opening inside `search.py` (prime away from blocked corner, toward the longer ray). Much smaller dev cost.

### Dev infrastructure — `tools/` (not shipped)
- `match_runner.py`: thin wrapper around `engine/run_local_agents.py` that spawns N parallel `play_game` processes via `multiprocessing.Pool`, collects `(winner, margin, timings, seed)` to a jsonl file.
- `tune.py`: coordinate descent over ~10 weights. Each evaluation = 20-game round-robin vs {previous self, Yolanda, a George-like scripted bot}. Run overnight on D6-D7. Use `cma` if installed, else coordinate descent.
- `regress.py`: must-pass smoke test — ≥95% wins vs Yolanda, no crashes, average play() under 3s.
- `logging` gated by env var `ALBRECHT_DEBUG=1` so shipped agent is silent.

## Critical files

- `engine/game/board.py` — reuse `get_valid_moves`, `forecast_move`, bitmasks, `reverse_perspective`.
- `engine/game/rat.py` — replicate emission model exactly in `belief.py` (NOISE_PROBS, DISTANCE_ERROR_PROBS/OFFSETS, 0-clamp behavior).
- `engine/game/enums.py` — `CARPET_POINTS_TABLE`, `RAT_BONUS=4`, `RAT_PENALTY=2`, `MAX_TURNS_PER_PLAYER=40`.
- `engine/gameplay.py` — match loop + timing semantics (reference only; don't import).
- `3600-agents/Yolanda/agent.py` — signature template.
- New: everything under `3600-agents/Albrecht/`.

## Implementation sequence (9 days)

### D1 — Skeleton + belief filter (2026-04-10)
- [x] Create `3600-agents/Albrecht/` directory with `__init__.py`, `agent.py`, `weights.py`.
- [x] `__init__.py`: `from .agent import PlayerAgent; from . import belief, search, eval`.
- [x] `agent.py`: class `PlayerAgent` with `__init__(board, transition_matrix, time_left)`, `play(board, sensor_data, time_left)`, `commentate()`. Initially returns a random valid move via `board.get_valid_moves()` (mirror Yolanda).
- [x] `t_precompute.py`: convert JAX T → numpy, compute `spawn_dist = T^1000 @ e_{(0,0)}` via repeated squaring (10 matmuls). Precompute `manhattan_lut[64,64]`, `dist_lik[observed, true]`, `noise_lik[cell_type, noise]` matching `engine/game/rat.py` constants exactly.
- [x] `belief.py`: `RatBelief` class with `predict()`, `update_sensor(noise, dist, worker_pos, board)`, `update_search(loc, hit)`, `clone()`, `argmax()`, `ev_best_search()`. State = `np.float32[64]`.
- [x] Wire belief into `agent.play`: predict → update_sensor from `sensor_data` → update_search from `board.opponent_search` and `board.player_search`.
- [x] Smoke test: `python3 engine/run_local_agents.py Albrecht Yolanda` — must run end-to-end without crashing for 20 games.
- [x] Verify belief stays normalized (sum ≈ 1.0) across a full game via assertion logging.
- [x] Verify belief's argmax correlates with the true rat position in game logs (dev-only check).

### D2 — Plain expectiminimax + v1 eval → beat George
- [x] `zobrist.py`: generate 64-bit random keys for (cell_type, cell_idx), (worker, loc), side_to_move. Build `hash(board)` function.
- [x] `search.py`: `Searcher` class with iterative deepening expectiminimax + alpha-beta, no TT yet. Handles move generation via `board.get_valid_moves()`, state transitions via `board.forecast_move()`. Returns `(best_move, best_value)`.
- [x] `eval.py` v1: weighted sum of `score_delta + carpet_potential(my) - carpet_potential(opp)`. `carpet_potential` walks the 4 rays from worker using `_primed_mask` shifts, returns max `CARPET_POINTS_TABLE` value achievable.
- [x] Integrate into `agent.play`: call searcher with per-turn budget `(time_left() - 15) / turns_remaining`.
- [x] Add fallback: wrap search in try/except; on any exception return any `board.get_valid_moves()[0]`.
- [x] Smoke test: Albrecht vs Yolanda → 100% win rate over 5 games (exceeds 95% target).
- [x] **Milestone: beat George.** George stub in `3600-agents/George/`. 100% win rate over 5 games (exceeds 60% target).

### D3 — TT + full heuristic → beat Albert (2026-04-13)
- [x] `search.py`: add Zobrist TT (`dict[int, TTEntry]` with depth, value, flag, best_move). Replace-by-depth policy. Size cap 2^18.
- [x] PV move ordering: TT best move first, then carpet (desc by roll score), then prime, then plain, then search.
- [x] `eval.py` v2: add `future_carpet_potential` (2-step prime lookahead with `γ=0.6` discount), `mobility` (2-radius BFS on bitboard), `setup_distance`, `dead_prime_penalty`.
- [x] Tune initial weights by hand — bumped CARPET_POTENTIAL_W 0.8→0.9, MOBILITY_W 0.1→0.15, SETUP_DISTANCE_W -0.05→-0.08.
- [x] Performance check: depth 9 in 2.7s on mid-game board (well above ≥4 target). TT providing ~6% hit rate.
- [x] **Milestone: beat Albert 100% (3/3 games).** `tools/albert_stub.py` + `3600-agents/Albert/` created. Also beats George 100% and Yolanda 100%.

### D4 — Belief-integrated search + opponent folding → beat Carrie
- [ ] `belief.py`: apply `update_search` to opponent's previous turn search (from `board.opponent_search`) BEFORE sensor update. Confirm via logging that miss zeros out belief at that cell.
- [ ] `search.py`: treat `Move.search(c)` as a chance node inside expectiminimax. Compute `p = belief[idx(c)]`; child value = `p*(4 + V_hit) + (1-p)*(-2 + V_miss)` where `V_hit` uses belief reset to `spawn_dist` and `V_miss` uses belief with `b[idx(c)]=0, renormalize`. Prune `SEARCH` generation to top-3 belief cells + any cell with `b>0.15`.
- [ ] `eval.py` v3: add `search_ev_best`, `belief_entropy`, `opponent_disruption`, `time_pressure`, `blocked_corner_awareness`. Pass belief into eval.
- [ ] Smoke test: Albrecht vs Albrecht-D3 snapshot → ≥70% win rate.
- [ ] **Milestone: beat Carrie on bytefight.org in ≥3/5 scrimmage games.** (Upload early — don't wait until D9.)

### D5 — Hardening + safety nets
- [ ] Time budget clamps: hard cutoff at 90% of per-turn budget via `time.monotonic()`; persistent wall-clock guard at `time_left() < 20` → emergency shallow search only.
- [ ] OOM safety: TT size hard cap (LRU eviction), periodic `gc.collect()` at turn boundaries, watchdog on RSS via `resource.getrusage` (log only; don't act).
- [ ] Exception fallback at every level: search → return shallowest best; shallowest → any valid move; all-fail → `Move.plain(Direction.UP)` (engine will detect invalid and we lose, but at least no crash).
- [ ] `commentate()`: return a fun one-liner (e.g., "gg, the rat never stood a chance").
- [ ] Stress test: run 50 consecutive games vs Yolanda without any crash or timeout.
- [ ] Test with `limit_resources=True` via a custom `gameplay.play_game` wrapper to catch seccomp issues early.

### D6–D7 — Tuning infra + overnight self-play
- [ ] `tools/match_runner.py`: `multiprocessing.Pool` wrapper around `engine.gameplay.play_game` (import as library, not via subprocess). Collect `(winner, a_points, b_points, turns, wall_time)` to JSONL. Support `N_parallel`, `N_games`, `agent_a`, `agent_b` args.
- [ ] `tools/tune.py`: coordinate descent over the ~10 weights in `weights.py`. Each evaluation = 20 games vs frozen baselines (`Albrecht-D5`, `Yolanda`, `george_stub`, `albert_stub`). Fitness = `0.6*winrate + 0.4*margin_norm`. Checkpoint best weights every step.
- [ ] `tools/regress.py`: hard smoke test — ≥95% vs Yolanda, no regressions >5 pp vs last committed weights.
- [ ] Kick off overnight tuning run (D6 night → D7 morning). Target ≥200 evaluations.
- [ ] Review tuning log on D7; freeze best weights into `weights.py`.
- [ ] Re-test against Carrie on bytefight.org to confirm improvement.

### D8 — Endgame solver + opening heuristic
- [ ] `endgame.py`: exhaustive alpha-beta for last `min(turns_left*2, 6)` plies. Exact terminal score delta evaluation. Reuse TT from main search (seed entries).
- [ ] Search dispatcher: when `turns_left ≤ 3` (per worker), call endgame solver instead of heuristic search.
- [ ] Opening heuristic in `search.py`: first 3 turns use a scripted rule — prime step toward the longest empty ray from spawn, away from blocked corners. Fall through to search if no good candidate.
- [ ] Final weight freeze: run `tune.py` one more pass (5 hours) starting from D7 weights for fine-tuning with endgame enabled.
- [ ] Regression gate: ≥95% vs Yolanda, ≥70% vs Albrecht-D5 snapshot, ≥55% vs Carrie.

### D9 — Submission
- [ ] Strip dev files: `tools/` directory, any `.pyc`, `matches/`, `__pycache__/`, debug logs.
- [ ] Verify `3600-agents/Albrecht/` contains only: `__init__.py`, `agent.py`, `belief.py`, `t_precompute.py`, `search.py`, `eval.py`, `zobrist.py`, `endgame.py`, `weights.py`, optionally `opponent_model.py`.
- [ ] Verify zip size ≤200 MB; typical expected size <2 MB (no NN weights).
- [ ] Dry-run the exact zip on `limit_resources=True` path; confirm init <10s, first move <10s.
- [ ] Upload `Albrecht.zip` to bytefight.org; run at least 5 scrimmages vs Carrie to confirm ELO on the real grader.
- [ ] Confirm team registered with correct emails per assignment §8.
- [ ] Final hard lockdown 24h before deadline; no more commits unless fixing crashes.

## Verification

- **Smoke test**: `python3 engine/run_local_agents.py Albrecht Yolanda` (and reverse order) — must win ≥95% over 20 games.
- **Regression tests**: `python3 3600-agents/Albrecht/tools/regress.py` — vs prior committed version, no win-rate regression >5%.
- **Performance**: assert average `play()` time < 3 s, max < 10 s, total per-game < 220 s (leaves 20 s buffer).
- **Sandbox simulation**: run with `limit_resources=True` path via `gameplay.play_game` to confirm no seccomp / rlimit issues.
- **End-to-end against ELO ladder**: target Carrie on bytefight.org scrimmage early (D5) and iterate from empirical match logs saved to `3600-agents/matches/`.

## Decisions locked in

- **Agent name:** `Albrecht` → directory `3600-agents/Albrecht/`.
- **Risk profile:** Top-team contender. Ship the safe Carrie-beater by D5, then layer on belief-integrated SEARCH, self-play tuning, and endgame exhaustive solver. No neural net (cut for time).
- **Dev priorities:** Full self-play tuning infrastructure with overnight runs. Parallel match runner + coordinate descent / CMA-ES over heuristic weights.
