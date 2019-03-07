# -*- coding: utf-8 -*-
import numpy as np
import SimpleITK as sitk
from os.path import isdir, join
from os import makedirs
from scipy.spatial.distance import cdist, pdist
from math import pi
import tifffile
from math import sqrt
DEBUG = True


def calc_max(pos):
    maks = 0
    for i, p in enumerate(pos):
        tmp = (pos - p)**2
        maks = max(maks, np.max(tmp[:,0] + tmp[:,1] + tmp[:,2]))
    return sqrt(maks)


def close_small_holes(image, max_hole_size):
    if image.dtype == np.bool:
        image = image.astype(np.uint8)
    if len(image.shape) == 2:
        rev_conn = sitk.ConnectedComponent(sitk.BinaryNot(sitk.GetImageFromArray(image)), True)
        return sitk.GetArrayFromImage(sitk.BinaryNot(sitk.RelabelComponent(rev_conn, max_hole_size)))
    for layer in image:
        rev_conn = sitk.ConnectedComponent(sitk.BinaryNot(sitk.GetImageFromArray(layer)), True)
        layer[...] = sitk.GetArrayFromImage(sitk.BinaryNot(sitk.RelabelComponent(rev_conn, max_hole_size)))
    return image


def opening(image, radius, minsize):
    rtype = "sitk"
    if isinstance(image, np.ndarray):
        rtype = "ndarray"
        image = sitk.GetImageFromArray(image)
    if isinstance(radius, (list, tuple)):
        radius = list(reversed(radius))
    eroded = sitk.GrayscaleErode(image, radius)
    conn = sitk.RelabelComponent(sitk.ConnectedComponent(eroded), 20)
    dilated = sitk.RelabelComponent(sitk.GrayscaleDilate(conn, radius), minsize)
    if rtype == "ndarray":
        return sitk.GetArrayFromImage(dilated)
    return dilated


def add_frame(image, frame):
    """
    :type image: np.ndarray
    :param image:
    :param frame:
    :return:
    """
    new_shape = []
    pos = []
    for x in image.shape[:3]:
        if x != 1:
            new_shape.append(x + 2*frame)
            pos.append(slice(frame, x+frame))
        else:
            new_shape.append(1)
            pos.append(slice(0, 1))
    new_shape = tuple(new_shape) + image.shape[3:]
    pos = tuple(pos)
    res = np.zeros(new_shape, dtype=image.dtype)
    res[pos] = image
    return res


def get_border(image):
    res_type = "sitk"
    if isinstance(image, np.ndarray):
        res_type = "ndarray"
        image = sitk.GetImageFromArray(image)
    border = sitk.LabelContour(image)
    if res_type == "ndarray":
        return sitk.GetArrayFromImage(border)
    return border


def remove_frame(image, frame):
    """
    :type image: np.ndarray
    :param image:
    :param frame:
    :return:
    """
    pos = []
    for x in image.shape:
        pos.append(slice(frame, x - frame))
    pos = tuple(pos)
    return np.copy(image[pos])


def cut_positive(image, with_position=False, dtype=None):
    positions = np.transpose(np.nonzero(image))
    lower = np.min(positions, 0)
    upper = np.max(positions, 0)+1
    pos = []
    for a, b in zip(lower, upper):
        pos.append(slice(a, b))
    pos = tuple(pos)
    if dtype is None:
        res = np.copy(image[pos])
    else:
        # noinspection PyUnresolvedReferences
        res = np.copy(image[pos]).astype(dtype)
    if with_position:
        return res, pos
    return res


def component_update(label_image, image, minsize):
    rtype = "sitk"
    if isinstance(label_image, np.ndarray):
        label_image = np.copy(label_image)
        rtype = "ndarray"
    if isinstance(label_image, sitk.Image):
        label_image = sitk.GetArrayFromImage(label_image)
    counter = label_image.max()+1
    values = range(1, counter)
    for num in values:
        # noinspection PyUnresolvedReferences
        part, pos = cut_positive(label_image == num, with_position=True, dtype=np.uint8)
        label_image[pos][part > 0] = 0
        # Check that should be split
        volume = np.count_nonzero(part) * 3
        expected_diam = (volume*(3.0/4) / pi)**(1.0/3)
        bord = get_border(part)
        positions = np.transpose(np.nonzero(bord))
        positions[:, 0] = positions[:, 0] * 3
        diam = calc_max(positions)

        if diam > 3 * expected_diam:
            print("Need split ", num)
            old_part = np.copy(part)
            part = opening(part, 5, minsize)

            if part.max() > 1:
                for i in range(2, part.max()+1):
                    if np.sum(part == i) > minsize/2:
                        label_image[pos][part == i] = counter
                        values.append(counter)
                        counter += 1
                part[part > 1] = 0
            else:
                part = old_part
        # TODO zrobić podział za dużych
        part = add_frame(part, 15).astype(np.uint8)
        # part = sitk.GetArrayFromImage(sitk.BinaryMorphologicalClosing(sitk.GetImageFromArray(part), 3))
        part = close_small_holes(part, 1000)
        part = sitk.GetArrayFromImage(sitk.BinaryMorphologicalClosing(sitk.GetImageFromArray(part), 8))
        part = close_small_holes(part, 1000)
        part = remove_frame(part, 15)
        label_image[pos][part > 0] = num

    ret = sitk.RelabelComponent(sitk.GetImageFromArray(label_image), minsize)
    # ret = sitk.GetImageFromArray(label_image)
    if rtype == "ndarray":
        return sitk.GetArrayFromImage(ret)
    return ret


def simply_segment(image, min_size, prefix, last_step=True, step_list=None, min_threshold=None):
    if isinstance(image, np.ndarray):
        image = sitk.GetImageFromArray(image)
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    thr = sitk.ThresholdMaximumConnectedComponents(image, min_size)
    if min_threshold is not None:
        stat = sitk.MinimumMaximumImageFilter()
        stat.Execute(image)
        thr2 = sitk.BinaryThreshold(image, min_threshold, stat.GetMaximum())
        thr = sitk.Minimum(thr, thr2)
    if DEBUG:
        test = sitk.GetArrayFromImage(sitk.Mask(image, thr))
        print("Threshold:", test[test > 0].min())
    # sitk.WriteImage(sitk.LabelToRGB(thr), "tmp1/" + prefix + "tmp0.tif")
    if isinstance(step_list, list):
        step_list.append(np.copy(sitk.GetArrayFromImage(thr)))
    thr2 = sitk.BinaryMorphologicalOpening(thr, 1)
    # thr2 = thr
    rm_small = sitk.RelabelComponent(sitk.ConnectedComponent(thr2, True), 1000)
    # sitk.WriteImage(sitk.LabelToRGB(rm_small), "tmp1/" + prefix + "tmp1.tif")
    if isinstance(step_list, list):
        step_list.append(np.copy(sitk.GetArrayFromImage(rm_small)))
    if not last_step and False:
        return sitk.RelabelComponent(rm_small, min_size)
    n_thr = close_small_holes(sitk.GetArrayFromImage(rm_small) > 0, 5000)
    fill1 = sitk.GetImageFromArray(n_thr)
    # sitk.WriteImage(sitk.LabelToRGB(fill1), "tmp1/" + prefix + "tmp2.tif")
    vot = sitk.VotingBinaryIterativeHoleFilling(fill1)
    # sitk.WriteImage(sitk.LabelToRGB(vot), "tmp1/" + prefix + "tmp3.tif")
    conn = sitk.RelabelComponent(sitk.ConnectedComponent(vot), min_size)
    return component_update(conn, image, min_size)


def re_segment(image, min_size, prefix, mask, good_list, min_threhold=None):
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    if isinstance(image, sitk.Image):
        image = sitk.GetArrayFromImage(image)
    if isinstance(mask, str):
        image = sitk.ReadImage(image)
    if isinstance(mask, sitk.Image):
        image = sitk.GetArrayFromImage(image)
    image = np.copy(image)
    good_list = set(good_list)
    for el in good_list:
        image[mask == el] = 0

    mask2 = np.copy(mask)
    i = 1
    for x in range(mask2.max()+1):
        if x in good_list:
            mask2[mask2 == x] = i
            i += 1
        else:
            mask2[mask2 == x] = 0
    mask = simply_segment(image, min_size, prefix, min_threshold=min_threhold)
    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask)
    mask[mask>0] += mask2.max()
    mask[mask2 > 0] = mask2[mask2 > 0]
    return sitk.GetImageFromArray(mask)


def simply_segment2(image, min_thresh, max_thresh, last_step=True):
    min_size = 80000
    if isinstance(image, np.ndarray):
        image = sitk.GetImageFromArray(image)
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    stat = sitk.MinimumMaximumImageFilter()
    stat.Execute(image)
    thr = sitk.DoubleThreshold(image,min_thresh, max_thresh,  stat.GetMaximum(), stat.GetMaximum()) # ThresholdMaximumConnectedComponents(image, min_size)
    if DEBUG:
        test = sitk.GetArrayFromImage(sitk.Mask(image, thr))
    n_thr = close_small_holes(sitk.GetArrayFromImage(thr), 1000)
    sitk.WriteImage(sitk.GetImageFromArray(n_thr), "tmp/tmp2.tif")
    fill1 = sitk.GetImageFromArray(n_thr)
    vot = sitk.VotingBinaryIterativeHoleFilling(fill1)
    conn = sitk.ConnectedComponent(vot)
    n_conn = sitk.GetArrayFromImage(sitk.RelabelComponent(conn, min_size))
    counter = n_conn.max()+1
    if not last_step:
        return sitk.RelabelComponent(conn, min_size)
    for num in range(1, n_conn.max()+1):
        # noinspection PyUnresolvedReferences
        part, pos = cut_positive(n_conn == num, with_position=True, dtype=np.uint8)
        part = add_frame(part, 15).astype(np.uint8)
        # TODO zrobić podział za dużych
        part = close_small_holes(part, 200)
        part = sitk.GetArrayFromImage(sitk.BinaryMorphologicalClosing(sitk.GetImageFromArray(part), 12))
        part = remove_frame(part, 15)
        n_conn[pos][part > 0] = num
    conn2 = sitk.GetImageFromArray(n_conn)
    return conn2


def cut_with_mask(mask, image, ignore=None, only=None):
    """
    :type mask: np.ndarray
    :type image: np.ndarray
    :type ignore: set
    :param mask:
    :param image:
    :return:
    """
    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask)
    if isinstance(mask, str):
        mask = tifffile.imread(mask)
    if isinstance(image, sitk.Image):
        image = sitk.GetArrayFromImage(image)
    if isinstance(image, str):
        image = tifffile.imread(image)
    if ignore is None:
        ignore = set()
    else:
        ignore = set(ignore)
    if len(image.shape) > len(mask.shape):
        apos = -1
        j = 0
        i = 0
        while i < len(mask.shape):
            if mask.shape[i] != image.shape[j]:
                if apos == -1:
                    apos = i
                else:
                    raise ValueError("Incompatible array shape")
                j+=1
            else:
                i+=1
                j+=1
        if apos != -1:
            image = image.swapaxes(apos,len(image.shape)-1)
            while apos < len(image.shape) - 2:
                image = image.swapaxes(apos,apos+1);
                apos += 1
    nonzero = np.bincount(mask.flat)
    res = []
    if only is None:
        to_cut_list = set(range(1, mask.max()+1)) - ignore
    else:
        to_cut_list = only
    for val in to_cut_list:
        if nonzero[val] == 0 or val in ignore:
            continue
        sub_mask, pos = cut_positive(mask == val, True)
        res2 = np.copy(image[pos])
        if len(res2.shape) > len(sub_mask.shape):

            for i in range(res2.shape[-1]):
                res2[..., i] *= sub_mask
        else:
            res2 *= sub_mask
        res.append((val, add_frame(res2, 3)))
    return res


def save_catted_list(images, path, prefix="", suffix=""):
    """
    :type images: list[tuple(int, np.ndarray)]
    :type path: str
    :param images:
    :param path:
    :param prefix:
    :return:
    """
    if not isdir(path):
        makedirs(path)
    for num, image in images:
        name = prefix + str(num) + suffix + ".tif"
        tifffile.imsave(join(path, name), image)

