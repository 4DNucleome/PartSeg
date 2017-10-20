from __future__ import division, print_function
from auto_fit import pointConverter as pc
import auto_fit as af
import numpy as np
from numpy import linalg
import logging
import sys
import h5py
import os
from math import asin, acos, pi, sqrt


def vector_cos(vec1, vec2):
    """
    Calculate value of cosinus between two vectors
    :type vec1: np.ndarray
    :type vec2: np.ndarray
    :return: float
    """
    return np.dot(vec1, vec2)/(linalg.norm(vec1) * linalg.norm(vec2))


def calculate_projection(vec1, vec2):
    """
    Calculate projection of vec1 on vec2
    :type vec1: np.ndarray
    :type vec2: np.ndarray:
    :return: np.ndarray
    """
    vec1_norm = linalg.norm(vec1)
    vec2_norm = linalg.norm(vec2)
    cos_val = np.dot(vec1, vec2)/(vec1_norm * vec2_norm)
    projection_length = cos_val * vec1_norm/vec2_norm
    return vec2 * projection_length


def calculate_orthogonal(vec1, vec2):
    """
    Calculate vec1 minus projection of vec1 on vec2
    :type vec1: np.ndarray
    :type vec2: np.ndarray:
    :return: np.ndarray
    """
    return vec1 - calculate_projection(vec1, vec2)


def coordinate_test1():
    vec1 = np.array([3, 0, 1])
    vec2 = np.array([2, 0, 0])
    np.testing.assert_almost_equal(vector_cos(vec1, vec1), 1)
    np.testing.assert_almost_equal(vector_cos(vec2, vec2), 1)
    np.testing.assert_almost_equal(vector_cos(calculate_projection(vec1, vec2), vec2), 1)
    np.testing.assert_almost_equal(vector_cos(calculate_projection(vec2, vec1), vec1), 1)
    np.testing.assert_almost_equal(vector_cos(calculate_orthogonal(vec1, vec2), vec2), 0)
    np.testing.assert_almost_equal(vector_cos(calculate_orthogonal(vec2, vec1), vec1), 0)


def get_rotation_matrix(image, voxel_size):
    """
    :type image: np.ndarray
    :type voxel_size: iterable
    :return: tuple[np.ndarray, np.ndarray]
    """
    img_eigen_vectors, img_eigen_values = af.find_density_orientation(image, voxel_size, cutoff=2000)
    np.testing.assert_almost_equal(np.sum(img_eigen_values), 1000)
    return linalg.eig(img_eigen_vectors)


def get_rotation_axis(rotation_matrix):
    eigen_val, eigen_vec = linalg.eig(rotation_matrix)
    for i, v in enumerate(eigen_val):
        if v.imag == 0:
            logging.info("reflection: {}".format(v.real < 0))
            return eigen_vec[:, i].real, v.real < 0

    return None


def get_rotation_angel(rotation_matrix, eigen_val=None, eigen_vec=None):
    if eigen_val is None or eigen_vec is None:
        eigen_val, eigen_vec = linalg.eig(rotation_matrix)
    for i, v in enumerate(eigen_val):
        if v.imag != 0:
            return asin(-v.imag) * 180 / pi, asin(v.imag) * 180 / pi, acos(v.real) * 180 / pi


def get_rotation_parameters(isometric_matrix):
    rotation_axis, reflection = get_rotation_axis(isometric_matrix)
    if reflection:
        print(linalg.det(isometric_matrix))
        isometric_matrix = np.dot(np.diag([-1, 1, 1]), isometric_matrix)
    print(linalg.det(isometric_matrix))
    angel = acos((np.sum(np.diag(isometric_matrix)) - 1) / 2) * 180 / pi
    square_diff = (isometric_matrix - isometric_matrix.T) ** 2
    denominator = sqrt(np.sum(square_diff) / 2)
    x = (isometric_matrix[2, 1] - isometric_matrix[1, 2]) / denominator
    y = (isometric_matrix[0, 2] - isometric_matrix[2, 0]) / denominator
    z = (isometric_matrix[1, 0] - isometric_matrix[0, 1]) / denominator
    return isometric_matrix, np.array((x, y, z)), angel


def rotation_test(cmap_path, out_path):
    h5_file = h5py.File(cmap_path, 'a')
    img = h5_file.get('/Chimera/image1/data_zyx').value
    img = np.swapaxes(img, 0, 2)  # change orientation from (z,y,z) to (x,y,z)
    grp = h5_file['/Chimera/image1']

    try:
        voxel_size = h5_file.get('/Chimera/image1/').attrs['step']
    except KeyError:
        voxel_size = np.array([1., 1., 1.])
    center_of_mass = af.density_mass_center(img, voxel_size)
    model_orientation, eigen_values = af.find_density_orientation(img, voxel_size, cutoff=2000)
    rotation_matrix, rotation_axis, angel = get_rotation_parameters(model_orientation.T)
    grp.attrs['rotation_axis'] = rotation_axis
    grp.attrs['rotation_angle'] = angel
    grp.attrs['origin'] = - np.dot(rotation_matrix, center_of_mass)
    # , reflection = get_rotation_axis(rotation_matrix)

    ort = calculate_orthogonal(center_of_mass, rotation_axis)
    proj = calculate_projection(center_of_mass, rotation_axis)
    logging.info("Center of mass: {}".format(center_of_mass))
    logging.info("Rotation axis: {}".format(rotation_axis))
    logging.info("Center of mass ort: {}".format(ort))
    logging.info("Center of mass projection: {}".format(calculate_projection(center_of_mass, rotation_axis)))
    pc.savePointsAsPdb([[0, 0, 0], center_of_mass], os.path.join(out_path,"mc.pdb"))
    pc.savePointsAsPdb([[0, 0, 0], rotation_axis*1000], os.path.join(out_path, "rot_ax.pdb"))
    pc.savePointsAsPdb([[0, 0, 0], ort], os.path.join(out_path, "ort.pdb"))
    pc.savePointsAsPdb([[0, 0, 0], proj], os.path.join(out_path, "proj.pdb"))
    pc.savePointsAsPdb([[0, 0, 0], np.dot(rotation_matrix, center_of_mass)], os.path.join(out_path, "rot_mc.pdb"))
    pc.savePointsAsPdb([[0, 0, 0], np.dot(rotation_matrix, ort)], os.path.join(out_path, "rot_ort.pdb"))
    pc.savePointsAsPdb([[0, 0, 0], np.dot(rotation_matrix, proj)], os.path.join(out_path, "rot_proj.pdb"))
    af.save_vectors_as_pdb(model_orientation, eigen_values, os.path.join(out_path, "model_orientation.pdb"),
                           center=center_of_mass)
    af.save_vectors_as_pdb(np.dot(rotation_matrix, model_orientation), eigen_values,
                           os.path.join(out_path, "rot_model_orientation.pdb"), full=True)
    h5_file.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.info("Epsilon: {}".format(sys.float_info.epsilon))
    coordinate_test1()
    af.save_vectors_as_pdb(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([900, 600, 400]),
                           os.path.join("res", "coord_orientation.pdb"), full=False)
    rotation_test("./res/nuc12syg1_rot.cmap", "res")
