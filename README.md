# Wormhole Transportation Network solver

An executable heuristic solver for [ESA's SpOC 2 challenge](https://optimize.esa.int/challenge/spoc-2-wormhole-transportation-network/About), using the database and original evaluation code included in this repository.

The solver finds routes for all 12 ships, synchronizes their expected arrivals within the database's time window, and minimizes the largest ship variance found during the search. It validates every candidate fleet before accepting it. This is a stochastic optimization method, with no guarantee of global optimality or feasibility on arbitrary replacement databases.

## Install and run

The solver uses only Python's standard library; no packages are required. With Python 3.11 or newer installed, run:

```powershell
python wormhole_solver.py
```

If you manage Python with [uv](https://docs.astral.sh/uv/), these commands create and use an isolated Python environment on Windows:

```powershell
uv venv --python 3.13
.venv/Scripts/python.exe wormhole_solver.py
```

On Linux/macOS, use `.venv/bin/python` in place of `.venv/Scripts/python.exe`.

The default database is `database (1).npz`, resolved relative to the script. The solver writes these files into `results/`:

| File | Contents |
| --- | --- |
| `solution.npy` | The 6,000-integer decision vector |
| `submission.json` | The challenge's JSON submission format |
| `report.json` | Fitness, constraint checks, arrivals, variances, jumps, complete paths and run settings |

To search more candidates with a reproducible random seed:

```powershell
.venv/Scripts/python.exe wormhole_solver.py --restarts 30 --attempts 168 --seed 123 --output results-long
```

The default is 8 restarts and 28 candidate walks per ship per restart. More search can improve the result but costs more time. Output is written only after finding a feasible solution; a failed search exits with an error. Each output folder is overwritten by a successful run, so use separate folders to compare runs. Nothing is submitted to ESA automatically.

The included `results/` was generated with seed 42, 8 restarts and 28 attempts per ship. The original ESA evaluator independently accepted it and agreed with all four entries of the standalone evaluator's fitness vector:

| Check | Result |
| --- | --- |
| Objective (lower is better) | 431.51475976628717 |
| Invalid origins | 0 |
| Invalid paths | 0 |
| Arrival spread | 0.9993119239807129 (limit: 1) |
| Maximum jumps per ship | 299 (limit: 500) |
| Recorded runtime | About 62 seconds on the development machine |

All 10 tests passed, including the optional comparisons against ESA's evaluator. Both saved submission formats were also validated with third-party packages disabled (`python -S`).

## Algorithm

1. Load the directed graph into compact sparse arrays using a numeric NPY/NPZ reader that does not allow pickle. Cast edge statistics and ship delays to float32, as the supplied evaluator does.
2. Choose a common arrival interval slightly narrower than the allowed window, leaving a margin for float32 rounding. Vary its centre between restarts.
3. Route the ships with the largest required time corrections first. Before routing each ship, build a minimum-variance suffix tree to the destination, excluding the other ships' reserved black holes. Dijkstra uses positive **variances** as weights; negative temporal means remain valid.
4. Construct randomized walks from the ship's allowed origins. Select edges by useful temporal offset per unit variance, increasing the required progress when the remaining jump budget is tight. Finish along the suffix tree once that gives an arrival in the common interval.
5. Reduce cost by replacing two-edge segments with cheaper one-edge or two-edge segments that preserve the arrival interval. Reserve the resulting route's nodes before constructing the next ship.
6. Evaluate the complete fleet with the challenge's actual minimax objective and retain the best feasible fleet across restarts.

All constructed routes are simple and share only the destination. This avoids exponential visit penalties, but restricts the search: an optimal solution might benefit from some sharing. The standalone evaluator supports shared nodes, repeated nodes and negative offsets, and applies the full chronological visit penalty, capped at `10^10`.

The encoded chromosome has one fixed-size block per ship. Each block contains the origin and intermediate nodes, followed by zero padding. The destination is implicit and is **not** stored as the final node of a block. A block can encode exactly `jump_limit` jumps, including the final edge to the destination. Paths in `report.json` include the destination for readability.

## Validate and test

Evaluate an existing result without running the optimizer:

```powershell
.venv/Scripts/python.exe wormhole_solver.py --validate results/solution.npy
.venv/Scripts/python.exe wormhole_solver.py --validate results/submission.json
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

For an independent comparison against the original ESA evaluator:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-validation.txt
.venv/Scripts/python.exe validate_official.py results/solution.npy
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

`validate_official.py` loads the supplied `.py.txt` file, skips only its final global instantiation (which references a missing default database directory), and instantiates the original class with the local database. It compares all four fitness-vector entries. The original NetworkX evaluator needs substantially more memory than the sparse solver for the two-million-edge graph.

Tests cover temporal compensation, directed-edge constraints, origins, chromosome encoding, jump limits, infeasible routes, malformed input, NPY compatibility and capped visit penalties. With validation dependencies installed, they also compare negative-time revisits, tied visits and the penalty cap against the original evaluator.
