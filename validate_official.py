"""Cross-check a saved solution against the evaluator originally supplied by ESA."""

import argparse
import ast
from pathlib import Path

import numpy as np

from wormhole_solver import Problem, evaluate


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('solution', nargs='?', type=Path, default=Path('results/solution.npy'))
    parser.add_argument('--database', type=Path, default=root / 'database (1).npz')
    args = parser.parse_args()
    source = next(root.glob('spoc-2-*.py.txt'))
    tree = ast.parse(source.read_text(encoding='utf-8'))
    tree.body = [statement for statement in tree.body if not (
        isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'udp' for target in statement.targets))]
    namespace = {}
    exec(compile(tree, str(source), 'exec'), namespace)
    udp = namespace['wormhole_traversal_udp'](database=args.database)
    x = np.load(args.solution, allow_pickle=False)
    expected = udp.fitness(x)
    actual = evaluate(Problem.load(args.database), x.tolist())['fitness']
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-8)
    if expected[1] != 0 or expected[2] != 0 or expected[3] > 0:
        raise SystemExit(f'Official evaluator rejected solution: {expected}')
    print(f'Official evaluator agrees; feasible. Fitness vector: {expected}')


if __name__ == '__main__':
    main()
