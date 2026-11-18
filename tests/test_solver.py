import array
import ast
import importlib.util
import io
import struct
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from npyio import read_npy, write_vector
from wormhole_solver import Problem, evaluate, solve


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def test_directory():
    # Windows managed sandboxes can reject tempfile's private (0700) directories.
    directory = ROOT / ('.test-' + uuid.uuid4().hex)
    directory.mkdir()
    try:
        yield directory
    finally:
        if directory.resolve().parent != ROOT:
            raise RuntimeError('Refusing to remove a directory outside the test root.')
        shutil.rmtree(directory)


def small_problem():
    return Problem(
        array.array('i', [1, 1, 2, 3, 3, 4, 5]),
        array.array('i', [4, 9, 9, 5, 9, 9, 9]),
        array.array('f', [.4, 0, 0, -.4, 0, 0, 0]),
        array.array('f', [.02, .001, .001, .02, .001, .01, .01]),
        [[1], [2], [3]], array.array('f', [-.8, 0, .8]), 9, 4, 1.)


class SolverTests(unittest.TestCase):
    def test_solver_compensates_offsets_and_satisfies_window(self):
        p = small_problem()
        x, report = solve(p, restarts=2, attempts=8, seed=4, progress=lambda _: None)
        self.assertTrue(report['feasible'])
        self.assertEqual(report['paths'], [[1, 4, 9], [2, 9], [3, 5, 9]])
        self.assertEqual(len(x), 12)
        self.assertLess(report['arrival_spread'], 1.)

    def test_encoding_destination_padding_and_exact_jump_limit(self):
        p = small_problem()
        p.jump_limit = 2
        paths = [[1, 4, 9], [2, 9], [3, 5, 9]]
        x = p.encode(paths)
        self.assertEqual(x, [1, 4, 2, 0, 3, 5])
        self.assertEqual(p.decode(x), paths)
        with self.assertRaises(ValueError):
            p.encode([[1, 4, 1, 9], [2, 9], [3, 5, 9]])

    def test_invalid_directed_edge_and_origin_are_reported(self):
        p = small_problem()
        report = evaluate(p, p.encode([[4, 1, 9], [2, 9], [3, 5, 9]]))
        self.assertFalse(report['feasible'])
        self.assertEqual(report['fitness'][1:3], [1, 1])

    def test_out_of_window_routes_are_infeasible(self):
        p = small_problem()
        report = evaluate(p, p.encode([[1, 9], [2, 9], [3, 9]]))
        self.assertFalse(report['feasible'])
        self.assertGreater(report['fitness'][3], 0)

    def test_malformed_chromosome_is_rejected(self):
        p = small_problem()
        for x in ([1], [0]*11+[.5], [0]*11+[float('nan')], [0]*11+[10], [0]*11+[1j]):
            with self.assertRaises(ValueError):
                p.decode(x)

    def test_unreachable_destination_never_produces_solution(self):
        p = Problem(array.array('i', [1, 3]), array.array('i', [2, 2]),
                    array.array('f', [0, 0]), array.array('f', [1, 1]),
                    [[1]], array.array('f', [5]), 3, 4, 1.)
        with self.assertRaises(RuntimeError):
            solve(p, restarts=1, attempts=2, progress=lambda _: None)

    def test_npy_round_trip_and_rejects_truncation(self):
        values = [1, 10000, 0, 5, -3]
        with test_directory() as directory:
            filename = Path(directory) / 'vector.npy'
            write_vector(filename, values)
            data = filename.read_bytes()
            actual, shape = read_npy(io.BytesIO(data))
            self.assertEqual(actual.tolist(), values)
            self.assertEqual(shape, (5,))
            with self.assertRaises(ValueError):
                read_npy(io.BytesIO(data[:-1]))

    def test_reads_big_endian_float_array(self):
        header = repr({'descr': '>f8', 'fortran_order': False, 'shape': (2,)}) + '\n'
        payload = b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header.encode() + struct.pack('>dd', 1.25, -.5)
        data, shape = read_npy(io.BytesIO(payload))
        self.assertEqual(data.tolist(), [1.25, -.5])
        self.assertEqual(shape, (2,))

    def test_shared_visits_and_penalty_cap(self):
        p = Problem(array.array('i', [1]), array.array('i', [2]),
                    array.array('f', [0]), array.array('f', [.003]),
                    [[1]]*12, array.array('f', [0]*12), 2, 2, 1.)
        report = evaluate(p, p.encode([[1, 2]]*12))
        self.assertTrue(report['feasible'])
        self.assertEqual(report['ship_variances'][10], report['ship_variances'][11])
        self.assertGreater(report['ship_variances'][10], report['ship_variances'][9])


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('numpy', 'networkx', 'matplotlib', 'loguru')),
                     'Install requirements-validation.txt to compare with ESA evaluator.')
class OfficialEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = next(ROOT.glob('spoc-2-*.py.txt'))
        tree = ast.parse(source.read_text(encoding='utf-8'))
        tree.body = [statement for statement in tree.body if not (
            isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == 'udp' for target in statement.targets))]
        namespace = {}
        exec(compile(tree, str(source), 'exec'), namespace)
        cls.udp_type = namespace['wormhole_traversal_udp']

    def test_negative_time_revisits_stable_ties_and_penalty_cap(self):
        import numpy as np
        edges = np.array([[1, 2], [1, 3], [3, 1]], dtype=np.int32)
        mv = np.array([[.04, .003], [-.2, .002], [.1, .001]])
        with test_directory() as directory:
            database = Path(directory) / 'fixture.npz'
            np.savez(database, edges=edges, meanvar=mv, origins=np.ones((12, 1), dtype=np.int32),
                     delays=np.array([0, 0, -.2, .2, 0, .5, -.7, 0, .1, -.2, 0, .3]),
                     destination=np.array(2), jump_limit=np.array(8), window=np.array(1.))
            p = Problem.load(database)
            udp = self.udp_type(database=database)
            for cycles in (0, 1, 2):
                paths = [[1] + [3, 1] * (cycles if ship % 2 else 0) + [2] for ship in range(12)]
                x = p.encode(paths)
                np.testing.assert_allclose(evaluate(p, x)['fitness'], udp.fitness(np.array(x)), rtol=1e-12, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
