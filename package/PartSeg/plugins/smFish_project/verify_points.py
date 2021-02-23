import numpy as np
from scipy.spatial.distance import cdist


def group_points(points: np.ndarray, max_dist=1):
    points = np.copy(points[:, 1:])
    points[:, 0] = np.round(points[:, 0])
    sort = np.argsort(points[:, 0])
    points = points[sort]
    max_val = points[-1, 0]
    prev_data = points[points[:, 0] == 0]
    point_groups = []
    index_info = {}
    for i in range(1, int(max_val + 1)):
        new_points = points[points[:, 0] == i]

        if new_points.size == 0 or prev_data.size == 0:
            index_info = {}
            for j, point in enumerate(new_points):
                index_info[j] = len(point_groups)
                point_groups.append([point])
            prev_data = new_points
            continue
        new_index_info = {}
        dist_array = cdist(new_points[:, 1:], prev_data[:, 1:])
        close_object = dist_array < max_dist
        consumed_set = set()
        close_indices = np.nonzero(close_object)
        for first, second in close_indices:
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
