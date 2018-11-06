from project_utils.algorithm_base import calculate_operation_radius
from project_utils.cmap_utils import CmapProfile
from project_utils.image_operations import RadiusType, gaussian
from .partseg_utils import HistoryElement, PartEncoder
import numpy as np
from tiff_image import Image, ImageWriter
import tarfile
from io import BytesIO, StringIO
import h5py
import typing
import os.path
import json
import datetime

# TODO add progress function to io


def get_tarinfo(name, buffer: typing.Union[BytesIO, StringIO]):
    tar_info = tarfile.TarInfo(name=name)
    buffer.seek(0)
    if isinstance(buffer, BytesIO):
        tar_info.size = len(buffer.getbuffer())
    else:
        tar_info.size = len(buffer.getvalue())
    tar_info.mtime = datetime.datetime.now().timestamp()
    return tar_info


def save_project(file_path: str, image: Image, segmentation: np.ndarray, full_segmentation: np.ndarray,
                 mask: typing.Union[np.ndarray, None], history: typing.List[HistoryElement],
                 algorithm_parameters: dict):
    ext = os.path.splitext(file_path)[1]
    if ext.lower() in ['.bz2', ".tbz2"]:
        tar_mod = 'w:bz2'
    else:
        tar_mod = 'w:gz'
    with tarfile.open(file_path, tar_mod) as tar:
        sek_dkt = {"segmentation": segmentation, "full_segmentation": full_segmentation}
        if mask is not None:
            sek_dkt["mask"] = mask
        seg_buff = BytesIO()
        np.savez(seg_buff, sek_dkt)
        tar_numpy = get_tarinfo("segmentation.npz", seg_buff)
        tar.addfile(tarinfo=tar_numpy, fileobj=seg_buff)
        image_buff = BytesIO()
        ImageWriter.save(image, image_buff)
        tar_image = get_tarinfo("image.tif", image_buff)
        tar.addfile(tarinfo=tar_image, fileobj=image_buff)
        para_str = json.dumps(algorithm_parameters, cls=PartEncoder)
        parameters_buff = BytesIO(para_str.encode('utf-8'))
        tar_algorithm = get_tarinfo("algorithm.json", parameters_buff)
        tar.addfile(tar_algorithm, parameters_buff)
        el_info = []
        for i, el in enumerate(history):
            el_info.append({"index": i, "algorithm_name": el.algorithm_name, "values": el.algorithm_values,
                            "mask_property": el.mask_property})
            el.arrays.seek(0)
            hist_info = get_tarinfo(f"history/arrays_{i}.npz", el.arrays)
            el.arrays.seek(0)
            tar.addfile(hist_info, el.arrays)
        if len(el_info) > 0:
            hist_str = json.dumps(el_info, cls=PartEncoder)
            hist_buff = BytesIO(hist_str.encode('utf-8'))
            tar_algorithm = get_tarinfo("history/history.json", hist_buff)
            tar.addfile(tar_algorithm, hist_buff)


def save_cmap(file: typing.Union[str, h5py.File], image: Image, segmentation: np.ndarray, cmap_profile: CmapProfile,
              metadata: typing.Optional[dict]):
    if segmentation is None or segmentation.max() == 0:
        raise ValueError("No segmentation")
    if isinstance(file, (str, BytesIO)):
        cmap_file = h5py.File(file)
    elif isinstance(file, h5py.File):
        cmap_file = file
    else:
        raise ValueError(f"Wrong type of file argument, type: {type(file)}")
    data = image.get_channel(cmap_profile.channel)
    if cmap_profile.gauss_radius > 0 and cmap_profile.gauss_type != RadiusType.NO:
        gauss_radius = calculate_operation_radius(cmap_profile.gauss_radius, image.spacing, cmap_profile.gauss_radius)
        layer = cmap_profile.gauss_type == RadiusType.R2D
        data = gaussian(data, gauss_radius, layer=layer)

    data[segmentation == 0] = 0
    grp = cmap_file.create_group('Chimera/image1')
    if cmap_profile.cut_obsolete_area:
        points = np.nonzero(segmentation)
        lower_bound = np.min(points, axis=1)
        upper_bound = np.max(points, axis=1)
        cut_img = np.zeros(upper_bound - lower_bound + [7, 7, 7], dtype=data.dtype)
        coord = tuple([slice(x, y) for x, y in zip(lower_bound, upper_bound)])
        cut_img[3:-3, 3:-3, 3:-3] = data[coord]
        data = cut_img
    z, y, x = data.shape
    data_set = grp.create_dataset("data_zyx", (z, y, x), dtype='f', compression="gzip")
    data_set[...] = data

    if metadata:
        meta_group = cmap_file.create_group('Chimera/image1/Statistics')
        for key, val in metadata.items():
            meta_group.attrs[key] = val

    grp = cmap_file['Chimera']
    grp.attrs['CLASS'] = np.string_('GROUP')
    grp.attrs['TITLE'] = np.string_('')
    grp.attrs['VERSION'] = np.string_('1.0')

    grp = cmap_file['Chimera/image1']
    grp.attrs['CLASS'] = np.string_('GROUP')
    grp.attrs['TITLE'] = np.string_('')
    grp.attrs['VERSION'] = np.string_('1.0')
    grp.attrs['step'] = np.array(image._image_spacing, dtype=np.float32)

    if isinstance(file, str):
        cmap_file.close()
    pass
