import numpy as np
from scipy.spatial.qhull import ConvexHull, QhullError

# this two functions are from
# https://stackoverflow.com/questions/37117878/generating-a-filled-polygon-inside-a-numpy-array/37123933#37123933


def check(p1, p2, idxs):
    """
    Uses the line defined by p1 and p2 to check array of
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    if p1[0] == p2[0]:
        max_col_idx = (idxs[0] - p1[0]) * idxs.shape[1]
        sign = np.sign(p2[1] - p1[1])
    else:
        max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
        sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign


def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    idxs = np.indices(shape)  # Create 3D array of indices
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k - 1], vertices[k], idxs)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 1

    return base_array


def _convex_fill(array: np.ndarray):
    if array.ndim != 2:
        raise ValueError("Convex fill need to be called on 2d array.")
    points = np.transpose(np.nonzero(array))
    try:
        convex = ConvexHull(points)
        convex_points = points[convex.vertices]
        convex.close()
        return create_polygon(array.shape, convex_points[::-1])
    except (QhullError, ValueError):
        return None


def convex_fill(array: np.ndarray):
    arr_shape = array.shape
    array = np.squeeze(array)
    if array.ndim not in [2, 3]:
        raise ValueError("Convex hull support only 2 and 3 dimension images")
    #  res = np.zeros(array.shape, array.dtype)
    components = np.bincount(array.flat)
    for i in range(1, components.size):
        if components[i] == 0:
            continue
        component: np.ndarray = array == i
        points = np.nonzero(component)
        if len(points) == 0 or len(points[0]) == 0:
            continue
        lower_bound = np.min(points, axis=1)
        upper_bound = np.max(points, axis=1)
        cut_area = tuple(slice(x, y + 1) for x, y in zip(lower_bound, upper_bound))
        if array.ndim == 3:
            cut_area = (slice(None),) + cut_area[1:]
        component = component[cut_area]
        if array.ndim == 2:
            res = _convex_fill(component)
            if res is None:
                continue
            array[cut_area][res > 0] = i
        elif array.ndim == 3:
            for j in range(lower_bound[0], upper_bound[0] + 1):
                res = _convex_fill(component[j])
                if res is None:
                    continue
                new_cut = (j,) + cut_area[1:]
                tmp = array[new_cut]
                tmp[res > 0] = i
                array[new_cut] = tmp
    return array.reshape(arr_shape)
