import h5py
import tempfile
import os
import tarfile
import sys
import numpy as np
import json
import logging
import SimpleITK as sitk
from copy import deepcopy
from enum import Enum
from math import acos, sqrt, pi

from backend import Settings, Segment, class_to_dict, calculate_statistic_from_image, get_segmented_data,\
    SegmentationProfile, SegmentationSettings

__author__ = "Grzegorz Bokota"


class MorphChange(Enum):
    no_morph = 1
    opening_morph = 2
    closing_morph = 3


class GaussUse(Enum):
    no_gauss = 1
    gauss_2d = 2
    gauss_3d = 3


def opening_smooth(volume_mask, radius=1):
    volume_mask = volume_mask.astype(np.uint8)
    sitk_image = sitk.GetImageFromArray(volume_mask)
    opening_image = sitk.BinaryMorphologicalOpening(sitk_image, radius)
    return sitk.GetArrayFromImage(opening_image)


def closing_smooth(volume_mask, radius=1):
    volume_mask = volume_mask.astype(np.uint8)
    sitk_image = sitk.GetImageFromArray(volume_mask)
    closing_image = sitk.BinaryMorphologicalClosing(sitk_image, radius)
    return sitk.GetArrayFromImage(closing_image)


ROTATION_MATRIX_DICT = {"x": np.diag([1, -1, -1]), "y": np.diag([-1, 1, -1]), "z": np.diag([-1, -1, 1])}


def save_to_cmap(file_path, settings, segment, gauss_type, with_statistics=True,
                 centered_data=True, morph_op=MorphChange.no_morph, scale_mass=(1,), rotate=None, with_cutting=True):
    """
    :type file_path: str
    :type settings: Settings
    :type segment: Segment
    :type gauss_type: GaussUse
    :type with_statistics: bool
    :type centered_data: bool
    :type morph_op: MorphChange
    :type scale_mass: (int)|list[int]
    :type rotate: str | None
    :type with_cutting: bool
    :return:
    """
    image = np.copy(settings.image)

    if gauss_type == GaussUse.gauss_2d or gauss_type == GaussUse.gauss_3d:
        image = segment.gauss_image
        logging.info("Gauss 2d")

    radius = 1
    if isinstance(morph_op, tuple) or isinstance(morph_op, list):
        radius = morph_op[1]
        morph_op = morph_op[0]
    if morph_op == MorphChange.no_morph:
        morph_fun = None
    elif morph_op == MorphChange.opening_morph:
        def morph_fun(img):
            logging.debug("Opening op radius {}".format(radius))
            return opening_smooth(img, radius)
    elif morph_op == MorphChange.closing_morph:
        def morph_fun(img):
            logging.debug("Closing op radius {}".format(radius))
            return closing_smooth(img, radius)
    else:
        logging.warning("Unknown morphological operation")
        morph_fun = None
    image, mask, noise_std = get_segmented_data(image, settings, segment, True, morph_fun, scale_mass[0])
    if gauss_type == GaussUse.gauss_3d:
        voxel = settings.spacing
        sitk_image = sitk.GetImageFromArray(image)
        sitk_image.SetSpacing(settings.spacing)
        image = sitk.GetArrayFromImage(sitk.DiscreteGaussian(sitk_image, max(voxel)))
        logging.info("Gauss 3d")

    points = np.nonzero(image)
    try:
        lower_bound = np.min(points, axis=1)
        upper_bound = np.max(points, axis=1)
    except ValueError:
        logging.error("No output")
        return

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    f = h5py.File(file_path, "w")
    grp = f.create_group('Chimera/image1')

    if with_cutting:
        cut_img = np.zeros(upper_bound - lower_bound + [3, 11, 11], dtype=image.dtype)
        coord = []
        for l, u in zip(lower_bound, upper_bound):
            coord.append(slice(l, u))
        pos = tuple(coord)
        cut_img[1:-2, 5:-6, 5:-6] = image[pos]
        z, y, x = cut_img.shape
        data_set = grp.create_dataset("data_zyx", (z, y, x), dtype='f')
        data_set[...] = cut_img
    else:
        z, y, x = image.shape
        data_set = grp.create_dataset("data_zyx", (z, y, x), dtype='f')
        data_set[...] = image
        cut_img = image

    if with_statistics:
        grp = f.create_group('Chimera/image1/Statistics')
        stat = calculate_statistic_from_image(image, mask, settings)
        for key, val in stat.items():
            grp.attrs[key] = val
        grp.attrs["Noise_std"] = noise_std

    # Just to satisfy file format
    grp = f['Chimera']
    grp.attrs['CLASS'] = np.string_('GROUP')
    grp.attrs['TITLE'] = np.string_('')
    grp.attrs['VERSION'] = np.string_('1.0')

    grp = f['Chimera/image1']
    grp.attrs['CLASS'] = np.string_('GROUP')
    grp.attrs['TITLE'] = np.string_('')
    grp.attrs['VERSION'] = np.string_('1.0')
    grp.attrs['step'] = np.array(settings.spacing, dtype=np.float32)

    if centered_data:
        swap_cut_img = np.swapaxes(cut_img, 0, 2)
        center_of_mass = density_mass_center(swap_cut_img, settings.spacing)
        model_orientation, eigen_values = find_density_orientation(swap_cut_img, settings.spacing, cutoff=1)
        if rotate is not None and rotate != "None":
            rotation_matrix, rotation_axis, angel = \
                get_rotation_parameters(np.dot(ROTATION_MATRIX_DICT[rotate], model_orientation.T))
        else:
            rotation_matrix, rotation_axis, angel = get_rotation_parameters(model_orientation.T)
        grp.attrs['rotation_axis'] = rotation_axis
        grp.attrs['rotation_angle'] = angel
        grp.attrs['origin'] = - np.dot(rotation_matrix, center_of_mass)

    data_set.attrs['CLASS'] = np.string_('CARRY')
    data_set.attrs['TITLE'] = np.string_('')
    data_set.attrs['VERSION'] = np.string_('1.0')

    f.close()


def save_to_project(file_path, settings, segment):
    """
    :type file_path: str
    :type settings: Settings
    :type segment: Segment
    :return:
    """
    folder_path = tempfile.mkdtemp()
    np.save(os.path.join(folder_path, "image.npy"), settings.image)
    np.save(os.path.join(folder_path, "draw.npy"), segment.draw_canvas)
    np.save(os.path.join(folder_path, "res_mask.npy"), segment.get_segmentation())
    if settings.image_clean_profile is not None:
        np.save(os.path.join(folder_path, "original_image.npy"), settings.original_image)
    if settings.mask is not None:
        np.save(os.path.join(folder_path, "mask.npy"), settings.mask)
    important_data = class_to_dict(settings, 'threshold_type', 'threshold_layer_separate', "threshold",
                                   "threshold_list", 'use_gauss', 'spacing', 'minimum_size', 'use_draw_result',
                                   "gauss_radius", "prev_segmentation_settings")
    if settings.image_clean_profile is not None:
        important_data["image_clean_profile"] = settings.image_clean_profile.__dict__
    else:
        important_data["image_clean_profile"] = None
    important_data["prev_segmentation_settings"] = deepcopy(important_data["prev_segmentation_settings"])
    for c, mem in enumerate(important_data["prev_segmentation_settings"]):
        np.save(os.path.join(folder_path, "mask_{}.npy".format(c)), mem["mask"])
        del mem["mask"]
    # image, segment_mask = get_segmented_data(settings.image, settings, segment)
    # important_data["statistics"] = calculate_statistic_from_image(image, segment_mask, settings)
    print(important_data)
    with open(os.path.join(folder_path, "data.json"), 'w') as ff:
        json.dump(important_data, ff)
    """if file_path[-3:] != ".gz":
        file_path += ".gz" """
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    ext = os.path.splitext(file_path)[1]
    if ext.lower() in ['.bz2', ".tbz2"]:
        tar = tarfile.open(file_path, 'w:bz2')
    else:
        tar = tarfile.open(file_path, 'w:gz')
    for name in os.listdir(folder_path):
        tar.add(os.path.join(folder_path, name), name)
    tar.close()


def load_project(file_path, settings, segment):
    """
    :type file_path: str
    :type settings: Settings | SegmentationSettings
    :type segment: Segment
    :return:
    """
    if sys.version_info.major == 2:
        def extract_numpy_file(name):
            return np.load(tar.extractfile(name))
    else:
        folder_path = tempfile.mkdtemp()

        def extract_numpy_file(name):
            tar.extract(name, folder_path)
            return np.load(os.path.join(folder_path, name))

    ext = os.path.splitext(file_path)[1]
    logging.debug("load_project extension: {}".format(ext))
    if ext.lower() in ['.bz2', ".tbz2"]:
        try:
            tar = tarfile.open(file_path, 'r:bz2')
        except tarfile.ReadError:
            tar = tarfile.open(file_path, 'r:gz')
    else:
        try:
            tar = tarfile.open(file_path, 'r:gz')
        except tarfile.ReadError:
            tar = tarfile.open(file_path, 'r:bz2')
    members = tar.getnames()
    json_val = tar.extractfile("data.json").read().decode("utf8")
    important_data = json.loads(json_val)
    image = extract_numpy_file("image.npy")
    draw = extract_numpy_file("draw.npy")
    if "mask.npy" in members:
        mask = extract_numpy_file("mask.npy")
    else:
        mask = None
    if "original_image.npy" in members:
        original_image = extract_numpy_file("original_image.npy")
        settings.image_clean_profile = SegmentationProfile(**important_data["image_clean_profile"])
    else:
        original_image = None
        settings.image_clean_profile = None
    settings.threshold = int(important_data["threshold"])
    settings.threshold_type = important_data["threshold_type"]
    settings.use_gauss = bool(important_data["use_gauss"])
    settings.spacing = \
        tuple(map(int, important_data["spacing"]))
    settings.voxel_size = settings.spacing
    settings.minimum_size = int(important_data["minimum_size"])
    try:
        if "use_draw_result" in important_data:
            settings.use_draw_result = int(important_data["use_draw_result"])
        else:
            settings.use_draw_result = int(important_data["use_draw"])
    except KeyError:
        settings.use_draw_result = False
    segment.protect = True
    settings.add_image(image, file_path, mask, new_image=False, original_image=original_image)
    segment.protect = False
    if important_data["threshold_list"] is not None:
        settings.threshold_list = map(int, important_data["threshold_list"])
    else:
        settings.threshold_list = []
    if "threshold_layer_separate" in important_data:
        settings.threshold_layer_separate = \
            bool(important_data["threshold_layer_separate"])
    else:
        settings.threshold_layer_separate = \
            bool(important_data["threshold_layer"])
    if "gauss_radius" in important_data:
        settings.gauss_radius = important_data["gauss_radius"]
    else:
        settings.gauss_radius = 1

    if "prev_segmentation_settings" in important_data:
        for c, mem in enumerate(important_data["prev_segmentation_settings"]):
            mem["mask"] = extract_numpy_file("mask_{}.npy".format(c))
        settings.prev_segmentation_settings = important_data["prev_segmentation_settings"]
    else:
        settings.prev_segmentation_settings = []
    settings.next_segmentation_settings = []
    segment.draw_update(draw)
    segment.threshold_updated()


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
    logging.info("\n============ Performing weighted PCA on image ============")

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
    logging.info("Eigen values0\n %s", str(values))
    logging.info('Eigen vectors0\n %s', str(vectors))
    sorted_values = sorted([(values[i], vectors[:, i]) for i in range(3)], key=lambda y: y[0], reverse=True)
    values = [x[0] for x in sorted_values]
    vectors = np.array([x[1] for x in sorted_values]).T
    logging.info("Eigen values\n %s", str(values))
    logging.info('Eigen vectors\n %s', str(vectors))
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
