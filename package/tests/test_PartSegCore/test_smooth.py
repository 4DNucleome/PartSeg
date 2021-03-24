import itertools

import numpy as np

from PartSegCore.segmentation.border_smoothing import IterativeVoteSmoothing, OpeningSmoothing, VoteSmoothing
from PartSegCore.segmentation.watershed import NeighType


class TestVoteSmoothing:
    def test_cube_sides(self):
        data = np.zeros((50, 50, 50), dtype=np.uint8)
        data[2:-2, 2:-2, 2:-2] = 1
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.sides, "support_level": 1})
        assert np.all(res == data)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.sides, "support_level": 3})
        assert np.all(res == data)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.sides, "support_level": 4})
        res2 = np.copy(data)
        for pos in itertools.product([2, -3], repeat=3):
            res2[pos] = 0
        assert np.all(res2 == res)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.sides, "support_level": 5})
        res2 = np.copy(data)
        for pos in itertools.permutations([2, 2, -3, -3, slice(2, -2)], 3):
            res2[pos] = 0
        assert np.all(res2 == res)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.sides, "support_level": 6})
        res2 = np.zeros(data.shape, dtype=data.dtype)
        res2[3:-3, 3:-3, 3:-3] = 1
        assert np.all(res2 == res)

    def test_cube_edges(self):
        data = np.zeros((50, 50, 50), dtype=np.uint8)
        data[2:-2, 2:-2, 2:-2] = 1
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.edges, "support_level": 1})
        assert np.all(res == data)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.edges, "support_level": 6})
        assert np.all(res == data)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.edges, "support_level": 7})
        res2 = np.copy(data)
        for pos in itertools.product([2, -3], repeat=3):
            res2[pos] = 0
        assert np.all(res2 == res)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.edges, "support_level": 9})
        assert np.all(res2 == res)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.edges, "support_level": 10})
        res2 = np.copy(data)
        for pos in itertools.permutations([2, 2, -3, -3, slice(2, -2)], 3):
            res2[pos] = 0
        assert np.all(res2 == res)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.edges, "support_level": 13})
        assert np.all(res2 == res)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.edges, "support_level": 14})
        res2 = np.zeros(data.shape, dtype=data.dtype)
        res2[3:-3, 3:-3, 3:-3] = 1
        assert np.all(res2 == res)

    def test_cube_vertex(self):
        data = np.zeros((50, 50, 50), dtype=np.uint8)
        data[2:-2, 2:-2, 2:-2] = 1
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.vertex, "support_level": 1})
        assert np.all(res == data)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.vertex, "support_level": 7})
        assert np.all(res == data)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.vertex, "support_level": 8})
        res2 = np.copy(data)
        for pos in itertools.product([2, -3], repeat=3):
            res2[pos] = 0
        assert np.all(res2 == res)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.vertex, "support_level": 11})
        assert np.all(res2 == res)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.vertex, "support_level": 12})
        res2 = np.copy(data)
        for pos in itertools.permutations([2, 2, -3, -3, slice(2, -2)], 3):
            res2[pos] = 0
        assert np.all(res2 == res)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.vertex, "support_level": 17})
        assert np.all(res2 == res)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.vertex, "support_level": 18})
        res2 = np.zeros(data.shape, dtype=data.dtype)
        res2[3:-3, 3:-3, 3:-3] = 1
        assert np.all(res2 == res)

    def test_square_sides(self):
        data = np.zeros((1, 50, 50), dtype=np.uint8)
        data[:, 2:-2, 2:-2] = 1
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.sides, "support_level": 1})
        assert np.all(res == data)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.sides, "support_level": 2})
        assert np.all(res == data)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.sides, "support_level": 3})
        res2 = np.copy(data)
        for pos in itertools.product([2, -3], repeat=2):
            res2[(0,) + pos] = 0
        assert np.all(res2 == res)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.sides, "support_level": 4})
        res2 = np.zeros(data.shape, dtype=data.dtype)
        res2[:, 3:-3, 3:-3] = 1
        assert np.all(res == res2)

    def test_square_edges(self):
        data = np.zeros((1, 50, 50), dtype=np.uint8)
        data[:, 2:-2, 2:-2] = 1
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.edges, "support_level": 1})
        assert np.all(res == data)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.edges, "support_level": 3})
        assert np.all(res == data)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.edges, "support_level": 4})
        res2 = np.copy(data)
        for pos in itertools.product([2, -3], repeat=2):
            res2[(0,) + pos] = 0
        assert np.all(res2 == res)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.edges, "support_level": 5})
        assert np.all(res2 == res)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.edges, "support_level": 6})
        res2 = np.zeros(data.shape, dtype=data.dtype)
        res2[:, 3:-3, 3:-3] = 1
        assert np.all(res == res2)

    def test_square_vertex(self):
        data = np.zeros((1, 50, 50), dtype=np.uint8)
        data[:, 2:-2, 2:-2] = 1
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.vertex, "support_level": 1})
        assert np.all(res == data)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.vertex, "support_level": 3})
        assert np.all(res == data)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.vertex, "support_level": 4})
        res2 = np.copy(data)
        for pos in itertools.product([2, -3], repeat=2):
            res2[(0,) + pos] = 0
        assert np.all(res2 == res)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.vertex, "support_level": 5})
        assert np.all(res2 == res)
        res = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.vertex, "support_level": 6})
        res2 = np.zeros(data.shape, dtype=data.dtype)
        res2[:, 3:-3, 3:-3] = 1
        assert np.all(res == res2)


def calc_cord(pos, sign, shift):
    return tuple(np.array(pos) + np.array(sign) * np.array(shift))


def generate_neighbour_sides(dist, ndim):
    return filter(lambda x: np.sum(x) <= dist, itertools.product(range(dist + 1), repeat=ndim))


def generate_neighbour_edge(dist, ndim):
    def sub_filter(arr):
        arr = np.sort(arr)
        val = 0
        mul = 1
        for el in reversed(arr):
            val += el * mul
            mul *= 2
        return val <= dist

    return filter(sub_filter, itertools.product(range(dist + 1), repeat=ndim))


class TestIterativeVoteSmoothing:
    def test_cube_sides_base(self):
        data = np.zeros((50, 50, 50), dtype=np.uint8)
        data[2:-2, 2:-2, 2:-2] = 1
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.sides, "support_level": 1, "max_steps": 1}
        )
        assert np.all(res == data)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.sides, "support_level": 3, "max_steps": 1}
        )
        assert np.all(res == data)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.sides, "support_level": 4, "max_steps": 1}
        )
        res2 = np.copy(data)
        for pos in itertools.product([2, -3], repeat=3):
            res2[pos] = 0
        assert np.all(res2 == res)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.sides, "support_level": 5, "max_steps": 1}
        )
        res2 = np.copy(data)
        for pos in itertools.permutations([2, 2, -3, -3, slice(2, -2)], 3):
            res2[pos] = 0
        assert np.all(res2 == res)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.sides, "support_level": 6, "max_steps": 1}
        )
        res2 = np.zeros(data.shape, dtype=data.dtype)
        res2[3:-3, 3:-3, 3:-3] = 1
        assert np.all(res2 == res)

    def test_cube_sides_iter(self):
        data = np.zeros((50, 50, 50), dtype=np.uint8)
        data[2:-2, 2:-2, 2:-2] = 1

        for i in range(2, 8):
            res = IterativeVoteSmoothing.smooth(
                data, {"neighbourhood_type": NeighType.sides, "support_level": 4, "max_steps": i}
            )
            res2 = np.copy(data)
            for pos in itertools.product([2, -3], repeat=3):
                sign = np.sign(pos)
                for shift in generate_neighbour_sides(i - 1, 3):
                    res2[calc_cord(pos, sign, shift)] = 0
            assert np.all(res2 == res), f"Fail  on step {i}"

        for i in range(2, 8):
            res = IterativeVoteSmoothing.smooth(
                data, {"neighbourhood_type": NeighType.sides, "support_level": 5, "max_steps": i}
            )
            res2 = np.copy(data)
            for pos in itertools.permutations([2, 2, -3, -3, slice(2, -2)], 3):
                res2[pos] = 0
            for ind in [0, 1, 2]:
                for pos in itertools.product([2, -3], repeat=2):
                    sign = np.sign(pos)
                    for shift in generate_neighbour_sides(i - 1, 2):
                        pos2 = list(calc_cord(pos, sign, shift))
                        pos2.insert(ind, slice(2, -2))
                        res2[tuple(pos2)] = 0
            assert np.all(res2 == res), f"Fail  on step {i}"

        for i in range(2, 8):
            res = IterativeVoteSmoothing.smooth(
                data, {"neighbourhood_type": NeighType.sides, "support_level": 6, "max_steps": i}
            )
            res2 = np.zeros(data.shape, dtype=data.dtype)
            shift = 2 + i
            p = slice(shift, -shift)
            res2[p, p, p] = 1
            assert np.all(res2 == res), f"Fail  on step {i}"

    def test_cube_edges_base(self):
        data = np.zeros((50, 50, 50), dtype=np.uint8)
        data[2:-2, 2:-2, 2:-2] = 1
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.edges, "support_level": 1, "max_steps": 1}
        )
        assert np.all(res == data)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.edges, "support_level": 6, "max_steps": 1}
        )
        assert np.all(res == data)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.edges, "support_level": 7, "max_steps": 1}
        )
        res2 = np.copy(data)
        for pos in itertools.product([2, -3], repeat=3):
            res2[pos] = 0
        assert np.all(res2 == res)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.edges, "support_level": 9, "max_steps": 1}
        )
        assert np.all(res2 == res)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.edges, "support_level": 10, "max_steps": 1}
        )
        res2 = np.copy(data)
        for pos in itertools.permutations([2, 2, -3, -3, slice(2, -2)], 3):
            res2[pos] = 0
        assert np.all(res2 == res)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.edges, "support_level": 13, "max_steps": 1}
        )
        assert np.all(res2 == res)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.edges, "support_level": 14, "max_steps": 1}
        )
        res2 = np.zeros(data.shape, dtype=data.dtype)
        res2[3:-3, 3:-3, 3:-3] = 1
        assert np.all(res2 == res)

    def test_cube_edge_iter(self):
        # This test can fail if IterativeVoting do not work correctly.
        data = np.zeros((50, 50, 50), dtype=np.uint8)
        data[2:-2, 2:-2, 2:-2] = 1
        res2 = np.copy(data)
        for pos in itertools.product([2, -3], repeat=3):
            res2[pos] = 0

        for i in range(2, 4):
            res = IterativeVoteSmoothing.smooth(
                data, {"neighbourhood_type": NeighType.edges, "support_level": 7, "max_steps": i}
            )
            assert np.all(res2 == res), f"Fail  on step {i}"
        for support_level in [8, 9, 10, 11, 12]:
            res2 = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.edges, "support_level": support_level})
            for i in range(2, 8):
                res = IterativeVoteSmoothing.smooth(
                    data, {"neighbourhood_type": NeighType.edges, "support_level": support_level, "max_steps": i}
                )
                res2 = VoteSmoothing.smooth(
                    res2, {"neighbourhood_type": NeighType.edges, "support_level": support_level}
                )
                assert np.all(res2 == res), f"Fail  on step {i} for support level {support_level}"

    def test_cube_vertex_base(self):
        data = np.zeros((50, 50, 50), dtype=np.uint8)
        data[2:-2, 2:-2, 2:-2] = 1
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.vertex, "support_level": 1, "max_steps": 1}
        )
        assert np.all(res == data)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.vertex, "support_level": 7, "max_steps": 1}
        )
        assert np.all(res == data)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.vertex, "support_level": 8, "max_steps": 1}
        )
        res2 = np.copy(data)
        for pos in itertools.product([2, -3], repeat=3):
            res2[pos] = 0
        assert np.all(res2 == res)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.vertex, "support_level": 11, "max_steps": 1}
        )
        assert np.all(res2 == res)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.vertex, "support_level": 12, "max_steps": 1}
        )
        res2 = np.copy(data)
        for pos in itertools.permutations([2, 2, -3, -3, slice(2, -2)], 3):
            res2[pos] = 0
        assert np.all(res2 == res)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.vertex, "support_level": 17, "max_steps": 1}
        )
        assert np.all(res2 == res)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.vertex, "support_level": 18, "max_steps": 1}
        )
        res2 = np.zeros(data.shape, dtype=data.dtype)
        res2[3:-3, 3:-3, 3:-3] = 1
        assert np.all(res2 == res)

    def test_cube_vertex_iter(self):
        # This test can fail if IterativeVoting do not work correctly.
        data = np.zeros((50, 50, 50), dtype=np.uint8)
        data[2:-2, 2:-2, 2:-2] = 1

        for i in range(2, 4):
            res = IterativeVoteSmoothing.smooth(
                data, {"neighbourhood_type": NeighType.vertex, "support_level": 7, "max_steps": i}
            )
            assert np.all(data == res), f"Fail  on step {i}"
        for support_level in range(8, 19):
            res2 = VoteSmoothing.smooth(data, {"neighbourhood_type": NeighType.vertex, "support_level": support_level})
            for i in range(2, 8):
                res = IterativeVoteSmoothing.smooth(
                    data, {"neighbourhood_type": NeighType.vertex, "support_level": support_level, "max_steps": i}
                )
                res2 = VoteSmoothing.smooth(
                    res2, {"neighbourhood_type": NeighType.vertex, "support_level": support_level}
                )
                assert np.all(res2 == res), f"Fail  on step {i} for support level {support_level}"

    def test_square_sides_base(self):
        data = np.zeros((1, 50, 50), dtype=np.uint8)
        data[:, 2:-2, 2:-2] = 1
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.sides, "support_level": 1, "max_steps": 1}
        )
        assert np.all(res == data)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.sides, "support_level": 2, "max_steps": 1}
        )
        assert np.all(res == data)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.sides, "support_level": 3, "max_steps": 1}
        )
        res2 = np.copy(data)
        for pos in itertools.product([2, -3], repeat=2):
            res2[(0,) + pos] = 0
        assert np.all(res2 == res)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.sides, "support_level": 4, "max_steps": 1}
        )
        res2 = np.zeros(data.shape, dtype=data.dtype)
        res2[:, 3:-3, 3:-3] = 1
        assert np.all(res == res2)

    def test_square_sides_iterative(self):
        data = np.zeros((1, 50, 50), dtype=np.uint8)
        data[:, 2:-2, 2:-2] = 1
        for i in range(2, 4):
            res = IterativeVoteSmoothing.smooth(
                data, {"neighbourhood_type": NeighType.sides, "support_level": 3, "max_steps": i}
            )
            res2 = np.copy(data)
            for pos in itertools.product([2, -3], repeat=2):
                sign = np.sign(pos)
                for shift in generate_neighbour_sides(i - 1, 2):
                    res2[(0,) + calc_cord(pos, sign, shift)] = 0
            assert np.all(res == res2)

        for i in range(2, 8):
            res = IterativeVoteSmoothing.smooth(
                data, {"neighbourhood_type": NeighType.sides, "support_level": 4, "max_steps": i}
            )
            res2 = np.zeros(data.shape, dtype=data.dtype)
            shift = 2 + i
            p = slice(shift, -shift)
            res2[:, p, p] = 1
            assert np.all(res2 == res)

    def test_square_edges_base(self):
        data = np.zeros((1, 50, 50), dtype=np.uint8)
        data[:, 2:-2, 2:-2] = 1
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.edges, "support_level": 1, "max_steps": 1}
        )
        assert np.all(res == data)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.edges, "support_level": 3, "max_steps": 1}
        )
        assert np.all(res == data)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.edges, "support_level": 4, "max_steps": 1}
        )
        res2 = np.copy(data)
        for pos in itertools.product([2, -3], repeat=2):
            res2[(0,) + pos] = 0
        assert np.all(res2 == res)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.edges, "support_level": 5, "max_steps": 1}
        )
        assert np.all(res2 == res)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.edges, "support_level": 6, "max_steps": 1}
        )
        res2 = np.zeros(data.shape, dtype=data.dtype)
        res2[:, 3:-3, 3:-3] = 1
        assert np.all(res == res2)

    def test_square_edges_iterative(self):
        data = np.zeros((1, 50, 50), dtype=np.uint8)
        data[:, 2:-2, 2:-2] = 1

        res2 = np.copy(data)
        for pos in itertools.product([2, -3], repeat=2):
            res2[(0,) + pos] = 0
        for _ in range(2, 5):
            res = IterativeVoteSmoothing.smooth(
                data, {"neighbourhood_type": NeighType.edges, "support_level": 4, "max_steps": 1}
            )
            assert np.all(res2 == res)
        for i in range(2, 8):
            res = IterativeVoteSmoothing.smooth(
                data, {"neighbourhood_type": NeighType.edges, "support_level": 5, "max_steps": i}
            )
            res2 = np.copy(data)
            for pos in itertools.product([2, -3], repeat=2):
                sign = np.sign(pos)
                for shift in generate_neighbour_edge(i - 1, 2):
                    res2[(0,) + calc_cord(pos, sign, shift)] = 0
            assert np.all(res == res2)
        for i in range(2, 8):
            res = IterativeVoteSmoothing.smooth(
                data, {"neighbourhood_type": NeighType.edges, "support_level": 6, "max_steps": i}
            )
            res2 = np.zeros(data.shape, dtype=data.dtype)
            shift = 2 + i
            p = slice(shift, -shift)
            res2[:, p, p] = 1
            assert np.all(res2 == res)

    def test_square_vertex_base(self):
        data = np.zeros((1, 50, 50), dtype=np.uint8)
        data[:, 2:-2, 2:-2] = 1
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.vertex, "support_level": 1, "max_steps": 1}
        )
        assert np.all(res == data)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.vertex, "support_level": 3, "max_steps": 1}
        )
        assert np.all(res == data)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.vertex, "support_level": 4, "max_steps": 1}
        )
        res2 = np.copy(data)
        for pos in itertools.product([2, -3], repeat=2):
            res2[(0,) + pos] = 0
        assert np.all(res2 == res)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.vertex, "support_level": 5, "max_steps": 1}
        )
        assert np.all(res2 == res)
        res = IterativeVoteSmoothing.smooth(
            data, {"neighbourhood_type": NeighType.vertex, "support_level": 6, "max_steps": 1}
        )
        res2 = np.zeros(data.shape, dtype=data.dtype)
        res2[:, 3:-3, 3:-3] = 1
        assert np.all(res == res2)

    def test_square_vertex_iterative(self):
        data = np.zeros((1, 50, 50), dtype=np.uint8)
        data[:, 2:-2, 2:-2] = 1

        res2 = np.copy(data)
        for pos in itertools.product([2, -3], repeat=2):
            res2[(0,) + pos] = 0
        for _ in range(2, 5):
            res = IterativeVoteSmoothing.smooth(
                data, {"neighbourhood_type": NeighType.vertex, "support_level": 4, "max_steps": 1}
            )
            assert np.all(res2 == res)
        for i in range(2, 8):
            res = IterativeVoteSmoothing.smooth(
                data, {"neighbourhood_type": NeighType.vertex, "support_level": 5, "max_steps": i}
            )
            res2 = np.copy(data)
            for pos in itertools.product([2, -3], repeat=2):
                sign = np.sign(pos)
                for shift in generate_neighbour_edge(i - 1, 2):
                    res2[(0,) + calc_cord(pos, sign, shift)] = 0
            assert np.all(res == res2)
        for i in range(2, 8):
            res = IterativeVoteSmoothing.smooth(
                data, {"neighbourhood_type": NeighType.vertex, "support_level": 6, "max_steps": i}
            )
            res2 = np.zeros(data.shape, dtype=data.dtype)
            shift = 2 + i
            p = slice(shift, -shift)
            res2[:, p, p] = 1
            assert np.all(res2 == res)


class TestOpeningSmooth:
    def test_cube(self):
        data = np.zeros((50, 50, 50), dtype=np.uint8)
        data[2:-2, 2:-2, 2:-2] = 1
        res = OpeningSmoothing.smooth(data, {"smooth_border_radius": 1})
        res2 = np.copy(data)
        for pos in itertools.product([2, -3], repeat=3):
            res2[pos] = 0
        assert np.all(res2 == res)

        res = OpeningSmoothing.smooth(data, {"smooth_border_radius": 2})
        res2 = np.copy(data)
        for pos in itertools.permutations([2, 2, -3, -3, slice(2, -2)], 3):
            res2[pos] = 0
        assert np.all(res2 == res)

    def test_sides(self):
        data = np.zeros((1, 50, 50), dtype=np.uint8)
        data[:, 2:-2, 2:-2] = 1
        res = OpeningSmoothing.smooth(data, {"smooth_border_radius": 1})
        assert np.all(data == res)
        res = OpeningSmoothing.smooth(data, {"smooth_border_radius": 2})
        res2 = np.copy(data)
        for pos in itertools.product([2, -3], repeat=2):
            res2[(0,) + pos] = 0
        assert np.all(res2 == res)
