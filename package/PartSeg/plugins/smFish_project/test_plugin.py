import numpy as np

from .verify_points import group_points


def test_grouping():
    data = np.array([[0, 1, 2, 1], [0, 1, 1, 4.1], [0, 5, 1, 3], [0, 2, 1, 4], [0, 2, 2, 2], [0, 2, 2, 1]])
    res = group_points(data)
    res = {tuple(tuple(x) for x in y) for y in res}
    assert res == {
        ((0.0, 1.0, 2.0, 1.0), (0.0, 2.0, 2.0, 1.0)),
        ((0.0, 1.0, 1.0, 4.1), (0.0, 2.0, 1.0, 4.0)),
        ((0.0, 2.0, 2.0, 2.0),),
        ((0.0, 5.0, 1.0, 3.0),),
    }
