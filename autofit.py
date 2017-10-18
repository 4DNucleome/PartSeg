import numpy as np
import logging
from math import acos, sqrt, pi
import scipy.spatial.distance as sp


def find_density_orientation(img, voxel_size, cutoff=1):
    """
       Identify axis of point set.

       Args:
           img (3D array): value in x, y, z
           voxel_size (len 3 vector): self explanatory
           cutoff (float): minimum value of value in image to take into account
       Returns:
           3x3 numpy array of eigen vectors
    """
    # logging.info("\n============ Performing weighted PCA on image ============")

    points_l = np.nonzero(np.array(img > cutoff))
    weights = img[points_l]
    points_l = np.transpose(points_l).astype(np.float64)
    points_l[:, 0] *= voxel_size[0]
    points_l[:, 1] *= voxel_size[1]
    points_l[:, 2] *= voxel_size[2]
    points = points_l.astype(np.float64)
    punkty_wazone = np.empty(points.shape)
    for i in range(3):
        punkty_wazone[:, i] = points[:, i] * weights
    mean = np.sum(punkty_wazone, axis=0)/np.sum(weights)
    points_shifted = points - mean
    wheighted_points_shifted = np.copy(points_shifted)
    for i in range(3):
        wheighted_points_shifted[:, i] *= weights
    cov = np.dot(wheighted_points_shifted.transpose(), points_shifted) * 1/(len(weights)-1)
    # cov variable is weighted covariance matrix
    values, vectors = np.linalg.eig(cov)
    # logging.info("Eigen values0\n %s", str(values))
    # logging.info('Eigen vectors0\n %s', str(vectors))
    sorted_values = sorted([(values[i], vectors[:, i]) for i in range(3)], key=lambda y: y[0], reverse=True)
    values = [x[0] for x in sorted_values]
    vectors = np.array([x[1] for x in sorted_values]).T
    # logging.info("Eigen values\n %s", str(values))
    # logging.info('Eigen vectors\n %s', str(vectors))
    w_n = values / np.sum(values) * 1000  # Drawing coordinates
    return vectors, w_n


def get_rotation_parameters(isometric_matrix):
    """
    If 3x3 isometric matrix is not rotation matrix
    function transform it into rotation matrix
    then calculate rotation axis and angel
    :param isometric_matrix: 3x3 np.ndarray with determinant equal 1 or -1
    :return: rotation_matrix, rotation axis, rotation angel
    """
    if np.linalg.det(isometric_matrix) < 0:
        isometric_matrix = np.dot(np.diag([-1, 1, 1]), isometric_matrix)
    angel = acos((np.sum(np.diag(isometric_matrix)) - 1) / 2) * 180 / pi
    square_diff = (isometric_matrix - isometric_matrix.T) ** 2
    denominator = sqrt(np.sum(square_diff) / 2)
    x = (isometric_matrix[2, 1] - isometric_matrix[1, 2]) / denominator
    y = (isometric_matrix[0, 2] - isometric_matrix[2, 0]) / denominator
    z = (isometric_matrix[1, 0] - isometric_matrix[0, 1]) / denominator
    return isometric_matrix, np.array((x, y, z)), angel


def density_mass_center(image, voxel_size=(1.0, 1.0, 1.0)):
    """
        Args:
            image: 3d numpy array

        Returns:
            x, y, z: three floats tuple with mass center coords
        :type image: np.ndarray
        :type voxel_size: tuple[float] | np.ndarray | list[float]
        :return np.ndarray

    """
    x_size, y_size, z_size = image.shape
    denominator = float(np.sum(image))
    m_x = np.sum(np.sum(image, axis=(1, 2)) * np.arange(x_size))
    m_y = np.sum(np.sum(image, axis=(0, 2)) * np.arange(y_size))
    m_z = np.sum(np.sum(image, axis=(0, 1)) * np.arange(z_size))
    return np.array([m_x/denominator, m_y/denominator, m_z/denominator]) * voxel_size


def calculate_density_momentum(image, voxel_size=np.array([1., 1., 1.]), mass_center=None):
    """Calculates image momentum."""
    if not mass_center:
        mass_center = density_mass_center(image, voxel_size)
    mass_center = np.array(mass_center)
    points = np.transpose(np.nonzero(np.zeros(image.shape, dtype=np.uint8)+1))
    points = points * voxel_size
    weights = np.squeeze(sp.cdist(points, np.array([mass_center]))**2)
    momentum = float(np.sum(weights * image.flatten()))
    return momentum