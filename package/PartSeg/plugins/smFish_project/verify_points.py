from itertools import product

import numpy as np
from magicgui import magic_factory
from napari import types
from napari.layers import Labels, Points
from scipy.spatial.distance import cdist


def group_points(points: np.ndarray, max_dist=1):
    points = np.copy(points)
    points[:, 1] = np.round(points[:, 1])
    sort = np.argsort(points[:, 1])
    points = points[sort]
    max_val = points[-1, 1]
    prev_data = points[points[:, 1] == 0]
    point_groups = []
    index_info = {}
    for i in range(1, int(max_val + 1)):
        new_points = points[points[:, 1] == i]

        if new_points.size == 0 or prev_data.size == 0:
            index_info = {}
            for j, point in enumerate(new_points):
                index_info[j] = len(point_groups)
                point_groups.append([point])
            prev_data = new_points
            continue
        new_index_info = {}
        dist_array = cdist(prev_data[:, 2:], new_points[:, 2:])
        close_object = dist_array < max_dist
        consumed_set = set()
        close_indices = np.nonzero(close_object)
        for first, second in zip(*close_indices):
            consumed_set.add(second)
            point_groups[index_info[first]].append(new_points[second])
            new_index_info[second] = index_info[first]
        for j, point in enumerate(new_points):
            if j in consumed_set:
                continue
            new_index_info[j] = len(point_groups)
            point_groups.append([point])
        prev_data = new_points
        index_info = new_index_info

    return point_groups


@magic_factory(info={"widget_type": "TextEdit"}, call_button=True)
def verify_segmentation(
    segmentation: Labels, points: Points, points_dist: int = 2, points_to_roi: int = 1, info: str = ""
) -> types.LayerDataTuple:
    labels = set(np.unique(segmentation.data))
    all_labels = len(labels)
    if 0 in labels:
        labels.remove(0)
    shift_array = np.array(
        [
            (0, 0, x, y)
            for x, y in product(range(-points_to_roi, points_to_roi + 1), repeat=2)
            if x ** 2 + y ** 2 <= points_to_roi ** 2
        ]
    )
    print(shift_array)
    points_grouped = group_points(points.data, points_dist)
    matched_points = [False for _ in points_grouped]
    for i, points_group in enumerate(points_grouped):
        for point in points_group:
            coords = (shift_array + point.astype(np.int16)).astype(np.int16)
            values = segmentation.data[tuple(coords.T)]
            for value in values:
                if value > 0:
                    if value in labels:
                        labels.remove(value)
                    matched_points[i] = True

    verify_segmentation.info.value = (
        f"matched {np.sum(matched_points)} of {len(matched_points)}"
        f"\nconsumed {all_labels - len(labels)} of {all_labels} segmentation components"
    )
    res = []
    for ok, points_group in zip(matched_points, points_grouped):
        if not ok:
            res.extend(points_group)
    if res:
        return np.array(res), {"name": "Missed points", "scale": points.scale}, "points"
    else:
        return None, {"name": "Missed points", "scale": points.scale}, "points"
