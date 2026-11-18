"""Dependency-free multistart routing for ESA's SpOC 2 wormhole challenge."""

from __future__ import annotations

import argparse
import array
import bisect
import heapq
import json
import math
import random
import struct
import time
import zipfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from npyio import read_npy, write_vector
from routing import _construct, _improve_route


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


@dataclass
class Problem:
    sources: array.array
    targets: array.array
    means: array.array
    variances: array.array
    origins: list[list[int]]
    delays: array.array
    destination: int
    jump_limit: int
    window: float

    @classmethod
    def load(cls, filename):
        with zipfile.ZipFile(filename) as archive:
            def read(name):
                with archive.open(name + '.npy') as stream:
                    return read_npy(stream)
            edges, edge_shape = read('edges')
            meanvar, mv_shape = read('meanvar')
            origins, origin_shape = read('origins')
            delays, delay_shape = read('delays')
            destination, _ = read('destination')
            jump_limit, _ = read('jump_limit')
            window, _ = read('window')
        if (len(edge_shape) != 2 or edge_shape[1] != 2 or edge_shape != mv_shape
                or len(origin_shape) != 2 or delay_shape != (origin_shape[0],)):
            raise ValueError('Unexpected database array shapes.')
        order = sorted(range(edge_shape[0]), key=lambda e: (edges[2*e], edges[2*e+1]))
        p = cls(array.array('i', (edges[2*e] for e in order)),
                array.array('i', (edges[2*e+1] for e in order)),
                array.array('f', (meanvar[2*e] for e in order)),
                array.array('f', (meanvar[2*e+1] for e in order)),
                [origins[i*origin_shape[1]:(i+1)*origin_shape[1]].tolist() for i in range(origin_shape[0])],
                array.array('f', delays), int(destination[0]), int(jump_limit[0]), float(window[0]))
        if (p.jump_limit < 1 or p.window <= 0 or not math.isfinite(p.window)
                or not p.sources or not p.delays
                or min(p.sources) < 1 or min(p.targets) < 1
                or any(not math.isfinite(v) or v <= 0 for v in p.variances)
                or any(not math.isfinite(m) for m in p.means)
                or any(not math.isfinite(d) for d in p.delays)
                or any(not row for row in p.origins)
                or not 1 <= p.destination <= p.n_nodes
                or any(not 1 <= node <= p.n_nodes for row in p.origins for node in row)):
            raise ValueError('Invalid database parameters, nodes, delays or edge statistics.')
        return p

    @property
    def n_ships(self):
        return len(self.delays)

    @cached_property
    def n_nodes(self):
        return max(max(self.sources), max(self.targets))

    def encode(self, paths):
        """Complete paths include the destination; encoded blocks omit it."""
        if len(paths) != self.n_ships:
            raise ValueError('Wrong number of paths.')
        chromosome = []
        for path in paths:
            if len(path) == 0 or path[-1] != self.destination:
                raise ValueError('Every path must end at the destination.')
            if len(path) - 1 > self.jump_limit:
                raise ValueError('Path exceeds jump limit; refusing to truncate it.')
            chromosome.extend(path[:-1])
            chromosome.extend([0] * (self.jump_limit - len(path) + 1))
        return chromosome

    def decode(self, chromosome):
        if len(chromosome) != self.n_ships * self.jump_limit:
            raise ValueError('Chromosome has the wrong length.')
        n_nodes = self.n_nodes
        if any(not isinstance(v, (int, float)) or isinstance(v, bool)
               or not math.isfinite(v) or v != int(v) or not 0 <= v <= n_nodes for v in chromosome):
            raise ValueError('Chromosome must contain integer node IDs within bounds.')
        paths = []
        for ship in range(self.n_ships):
            row = [int(v) for v in chromosome[ship*self.jump_limit:(ship+1)*self.jump_limit]]
            end = row.index(0) if 0 in row else len(row)
            paths.append(row[:end] + [self.destination])
        return paths


class Network:
    def __init__(self, problem):
        self.problem = problem
        self.n_nodes = problem.n_nodes
        self.targets = problem.targets
        self.indptr = array.array('i', [0]) * (self.n_nodes + 2)
        self.reverse_ptr = array.array('i', [0]) * (self.n_nodes + 2)
        for source, target in zip(problem.sources, problem.targets):
            self.indptr[source+1] += 1
            self.reverse_ptr[target+1] += 1
        for node in range(1, self.n_nodes + 2):
            self.indptr[node] += self.indptr[node-1]
            self.reverse_ptr[node] += self.reverse_ptr[node-1]
        self.reverse_edges = array.array('i', [0]) * len(problem.sources)
        cursor = self.reverse_ptr[:]
        for edge, target in enumerate(problem.targets):
            self.reverse_edges[cursor[target]] = edge
            cursor[target] += 1

    def edge(self, source, target):
        if not 1 <= source <= self.n_nodes:
            return None
        lo, hi = self.indptr[source:source+2]
        edge = bisect.bisect_left(self.targets, target, lo, hi)
        return edge if edge < hi and self.targets[edge] == target else None

    def suffixes(self, blocked):
        """Dijkstra on the reversed graph, weighted by positive variance."""
        p = self.problem
        distance = [math.inf] * (self.n_nodes + 1)
        successor = [-1] * (self.n_nodes + 1)
        distance[p.destination] = 0.
        heap = [(0., p.destination)]
        while heap:
            cost, target = heapq.heappop(heap)
            if cost != distance[target]:
                continue
            for offset in range(self.reverse_ptr[target], self.reverse_ptr[target+1]):
                edge = self.reverse_edges[offset]
                source = p.sources[edge]
                if blocked[source]:
                    continue
                candidate = cost + p.variances[edge]
                if candidate < distance[source]:
                    distance[source] = candidate
                    successor[source] = target
                    heapq.heappush(heap, (candidate, source))
        means = [0.] * (self.n_nodes + 1)
        hops = [p.jump_limit + 1] * (self.n_nodes + 1)
        hops[p.destination] = 0
        for node in sorted(range(self.n_nodes+1), key=distance.__getitem__):
            target = successor[node]
            if target >= 0:
                edge = self.edge(node, target)
                means[node] = p.means[edge] + means[target]
                hops[node] = hops[target] + 1
        return successor, means, hops


def evaluate(problem, chromosome, network=None):
    """Match the supplied UDP's float32 times, visit ordering and variance cap."""
    network = network or Network(problem)
    paths = problem.decode(chromosome)
    visits, edge_paths, arrivals = {}, [], []
    origin_errors = path_errors = 0
    for ship, path in enumerate(paths):
        origin_errors += int(path[0] not in problem.origins[ship])
        clock, path_mean, edge_path = problem.delays[ship], 0., []
        valid = True
        for jump, (source, target) in enumerate(zip(path, path[1:])):
            edge = network.edge(source, target)
            if edge is None:
                valid = False
                break
            visits.setdefault(source, []).append((clock, ship, jump, edge))
            clock = f32(clock + problem.means[edge])
            path_mean = f32(path_mean + problem.means[edge])
            edge_path.append(edge)
        path_errors += int(not valid)
        visits.setdefault(path[-1], []).append((clock, ship, len(path)-1, None))
        arrivals.append(f32((path_mean if valid else ship*problem.window) + problem.delays[ship]))
        edge_paths.append(edge_path)
    penalties = {}
    for node_visits in visits.values():
        # Python's stable sort preserves ship/jump order for tied times.
        for rank, (_, ship, jump, edge) in enumerate(sorted(node_visits, key=lambda v: v[0])):
            if edge is not None:
                penalties[ship, jump] = 10. ** min(rank, 10)
    costs = [sum(f32(problem.variances[edge] * penalties[ship, jump])
                 for jump, edge in enumerate(edges)) for ship, edges in enumerate(edge_paths)]
    spread = f32(max(arrivals) - min(arrivals))
    fitness = [max(costs)*10000, origin_errors, path_errors, f32(spread-problem.window)]
    return {'fitness': fitness, 'feasible': origin_errors == path_errors == 0 and fitness[3] <= 0,
            'arrivals': arrivals, 'arrival_spread': spread, 'ship_variances': costs,
            'jumps': [len(path)-1 for path in paths], 'paths': paths}


def solve(problem, restarts=8, attempts=28, seed=42, progress=print):
    network = Network(problem)
    rng = random.Random(seed)
    best = None
    margin = max(1e-4, problem.window * 2e-4)
    if 2*margin >= problem.window:
        raise ValueError('Window is too narrow for this solver\'s float32 safety margin.')
    initial_centre = (min(problem.delays)+max(problem.delays))/2
    for restart in range(restarts):
        centre = initial_centre + (0. if restart == 0 else rng.uniform(-.04, .04)*problem.window)
        lower, upper = centre-problem.window/2+margin, centre+problem.window/2-margin
        difficulty = [max(lower-d, d-upper) for d in problem.delays]
        priority = [d + rng.uniform(-.015, .015) for d in difficulty]
        order = sorted(range(problem.n_ships), key=lambda ship: -priority[ship])
        blocked = [False] * (network.n_nodes+1)
        paths = [None] * problem.n_ships
        for ship in order:
            successor, suffix_mean, suffix_hops = network.suffixes(blocked)
            path, _ = _construct(network.indptr, network.targets, problem.means,
                problem.variances, problem.origins[ship], problem.delays[ship],
                problem.destination, problem.jump_limit, lower, upper, blocked,
                successor, suffix_mean, suffix_hops, attempts, rng.randrange(2**31))
            if not path:
                progress(f'Restart {restart+1}: no route found for ship {ship+1}; trying another construction.')
                break
            path = _improve_route(path, network.indptr, network.targets, problem.means,
                                  problem.variances, problem.delays[ship], lower, upper, blocked)
            paths[ship] = path
            for node in path[:-1]:
                blocked[node] = True
        if any(path is None for path in paths):
            continue
        report = evaluate(problem, problem.encode(paths), network)
        if report['feasible'] and (best is None or report['fitness'][0] < best['fitness'][0]):
            best = report
            progress(f'Restart {restart+1}/{restarts}: fitness {report["fitness"][0]:.6f}; '
                     f'arrival spread {report["arrival_spread"]:.6f}; max jumps {max(report["jumps"])}')
    if best is None:
        raise RuntimeError('No feasible solution found. Increase --attempts or --restarts; no submission was written.')
    return problem.encode(best['paths']), best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path(__file__).with_name('database (1).npz'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--restarts', type=int, default=8)
    parser.add_argument('--attempts', type=int, default=28, help='Candidate walks per ship per restart.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validate', type=Path, help='Evaluate an existing .npy or submission JSON.')
    args = parser.parse_args()
    if args.restarts < 1 or args.attempts < 1:
        parser.error('--restarts and --attempts must be positive.')
    start = time.perf_counter()
    problem = Problem.load(args.database)
    if args.validate:
        if args.validate.suffix == '.npy':
            with args.validate.open('rb') as stream:
                data, shape = read_npy(stream)
            if shape != (problem.n_ships*problem.jump_limit,):
                raise ValueError('Expected a one-dimensional decision vector.')
            chromosome = data.tolist()
        else:
            chromosome = json.loads(args.validate.read_text(encoding='utf-8'))[0]['decisionVector']
        report = evaluate(problem, chromosome)
        print(json.dumps(report, indent=2))
        return 0 if report['feasible'] else 1
    chromosome, report = solve(problem, args.restarts, args.attempts, args.seed)
    report.update(seed=args.seed, restarts=args.restarts, attempts_per_ship=args.attempts,
                  elapsed_seconds=time.perf_counter()-start)
    args.output.mkdir(parents=True, exist_ok=True)
    write_vector(args.output / 'solution.npy', chromosome)
    (args.output / 'submission.json').write_text(json.dumps([{
        'decisionVector': chromosome, 'problem': 'wormhole-transportation-network',
        'challenge': 'spoc-2-wormhole-transportation-network'}], indent=2), encoding='utf-8')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Validated solution written to {args.output}; {report["elapsed_seconds"]:.1f}s.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
