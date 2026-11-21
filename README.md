# Sokoban solver — push-optimal, room by room

Entry for make.fun's "Build the best Sokoban solver, 21 November 2026" bounty.
`main.ts` (Node 22, no dependencies) reads every level under `/input/<id>/level.txt`
and writes `/output/<id>/solution.txt` with the player's moves in LURD.

## How it works

The generator lays one hallway across the board with one to four rooms off it,
each holding 3–4 boxes and as many goals, and a box pushed out of its room can
never come back — so every room is an independent problem and the rooms'
push-optima add up to the level's. The solver therefore:

1. **Finds the rooms** — it cuts the dead squares of the player's row (the
   hallway: a box there reaches no goal) out of the floor; each remaining part
   holding a box is a room.
2. **Solves each room with A\* over pushes** — a state is the boxes' squares
   plus the player's reachable region; the cost is pushes made and the estimate
   is the cheapest assignment of boxes to goals by floor distance (admissible:
   one push moves one box one square, so it cuts the estimate by at most one).
   Two deadlocks are pruned, both provably safe: a box on a square from which
   no sequence of pushes (ignoring the other boxes) reaches any goal, and a
   box frozen inside a 2×2 block of walls and boxes with a box off its goal.
   A plain push-BFS with the same pruning is the fallback when A\* exhausts
   its state budget.
3. **Walks between rooms** — the other rooms' boxes stand as walls while one
   room is solved, and the player walks on from where the last room left it.

Every budget is counted in states, never the clock, so two runs answer byte
for byte alike (verified: the kit's local grader reproduces the run).

## Results (kit's local grader, `--driver process`)

- Practice set: **score 1,000,000** — all 5 instances solved in the
  push-optimal count (1,000/1,000 each), ~0.4 s for the whole set.
- Three freshly drawn sets from the same generator (different seeds):
  **1,000,000 each**, each under 0.5 s.
- The house baseline (plain move-A\*) scores 200,000 on the practice set
  (only the first instance solved).

## License

MIT — see `LICENSE`.
