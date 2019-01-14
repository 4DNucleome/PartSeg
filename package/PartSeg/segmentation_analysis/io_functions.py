from pathlib import Path

from ..segmentation_analysis.analysis_utils import HistoryElement
from ..partseg_utils.channel_class import Channel
from ..partseg_utils.segmentation.algorithm_base import calculate_operation_radius
from ..partseg_utils.cmap_utils import CmapProfile
from ..partseg_utils.image_operations import RadiusType, gaussian
from ..partseg_utils.segmentation.algorithm_describe_base import Register, AlgorithmProperty
from ..partseg_utils.universal_const import UNITS_LIST, UNIT_SCALE
from .save_hooks import PartEncoder, part_hook
import numpy as np
from PartSeg.tiff_image import Image, ImageWriter, ImageReader
import tarfile
from io import BytesIO, TextIOBase, BufferedIOBase, RawIOBase, IOBase
import h5py
import typing
import os.path
import json
from functools import partial
from ..partseg_utils.io_utils import get_tarinfo, SaveBase
from .save_register import save_register


# TODO add progress function to io


def save_project(file_path: str, image: Image, segmentation: np.ndarray, full_segmentation: np.ndarray,
                 mask: typing.Optional[np.ndarray], history: typing.List[HistoryElement],
                 algorithm_parameters: dict):
    # TODO add support for binary objects
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
        np.savez(seg_buff, **sek_dkt)
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


class ProjectTuple(typing.NamedTuple):
    file_path: str
    image: Image
    segmentation: np.ndarray
    full_segmentation: np.ndarray
    mask: typing.Optional[np.ndarray]
    history: typing.List[HistoryElement]
    algorithm_parameters: dict


def load_project(
        file: typing.Union[str, tarfile.TarFile, TextIOBase, BufferedIOBase, RawIOBase, IOBase]) -> ProjectTuple:
    """Load project from archive"""
    if isinstance(file, tarfile.TarFile):
        tar_file = file
        file_path = ""
    elif isinstance(file, str):
        tar_file = tarfile.open(file)
        file_path = file
    elif isinstance(file, (TextIOBase, BufferedIOBase, RawIOBase, IOBase)):
        tar_file = tarfile.open(fileobj=file)
        file_path = ""
    else:
        raise ValueError(f"wrong type of file_ argument: {type(file)}")
    image_buffer = BytesIO()
    image_tar = tar_file.extractfile(tar_file.getmember("image.tif"))
    image_buffer.write(image_tar.read())
    image_buffer.seek(0)
    reader = ImageReader()
    image = reader.read(image_buffer)
    seg_tar = tar_file.extractfile(tar_file.getmember("segmentation.npz"))
    seg_buffer = BytesIO()
    seg_buffer.write(seg_tar.read())
    seg_buffer.seek(0)
    seg_dict = np.load(seg_buffer)
    if "mask" in seg_dict:
        mask = seg_dict["mask"]
    else:
        mask = None
    algorithm_str = tar_file.extractfile("algorithm.json").read()
    algorithm_dict = json.loads(algorithm_str, object_hook=part_hook)
    history = []
    try:
        history_buff = tar_file.extractfile(tar_file.getmember("history/history.json"))
        history_json = json.load(history_buff, object_hook=part_hook)
        for el in history_json:
            history_buffer = BytesIO()
            history_buffer.write(tar_file.extractfile(f"history/arrays_{el['index']}.npz").read())
            history_buffer.seek(0)
            history.append(HistoryElement(algorithm_name=el["algorithm_name"], algorithm_values=el["values"],
                                          mask_property=el["mask_property"], arrays=history_buffer))

    except KeyError:
        pass
    if isinstance(file, str):
        tar_file.close()
    return ProjectTuple(file_path, image, seg_dict["segmentation"], seg_dict["full_segmentation"], mask, history,
                        algorithm_dict)


def save_cmap(file: typing.Union[str, h5py.File], image: Image, segmentation: np.ndarray, cmap_profile: dict,
              metadata: typing.Optional[dict] = None):
    if segmentation is None or segmentation.max() == 0:
        raise ValueError("No segmentation")
    if isinstance(file, (str, BytesIO)):
        if isinstance(file, str) and os.path.exists(file):
            os.remove(file)
        cmap_file = h5py.File(file)
    elif isinstance(file, h5py.File):
        cmap_file = file
    else:
        raise ValueError(f"Wrong type of file argument, type: {type(file)}")
    data = image.get_channel(cmap_profile["channel"])
    """if cmap_profile.gauss_radius > 0 and cmap_profile.gauss_type != RadiusType.NO:
        gauss_radius = calculate_operation_radius(cmap_profile.gauss_radius, image.spacing, cmap_profile.gauss_radius)
        layer = cmap_profile.gauss_type == RadiusType.R2D
        data = gaussian(data, gauss_radius, layer=layer)"""

    data[segmentation == 0] = 0
    grp = cmap_file.create_group('Chimera/image1')
    if cmap_profile["clip"]:
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
    grp.attrs['step'] = np.array(image._image_spacing, dtype=np.float32)[::-1] * \
                        UNIT_SCALE[UNITS_LIST.index(cmap_profile["units"])]

    if isinstance(file, str):
        cmap_file.close()
    pass


class SaveProject(SaveBase):
    @classmethod
    def get_name(cls):
        return "Project (*.tgz *.tbz2 *.gz *.bz2)"

    @classmethod
    def get_short_name(cls):
        return "project"

    @classmethod
    def get_fields(cls):
        return []

    @classmethod
    def save(cls, save_location: typing.Union[str, BytesIO, Path], project_info: ProjectTuple, parameters: dict):
        save_project(save_location, project_info.image, project_info.segmentation, project_info.full_segmentation,
                     project_info.mask, project_info.history, project_info.algorithm_parameters)


class SaveCmap(SaveBase):
    @classmethod
    def get_name(cls):
        return "Chimera CMAP (*.cmap)"

    @classmethod
    def get_short_name(cls):
        return "cmap"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("channel", "Channel", 0, property_type=Channel),
                AlgorithmProperty("separated_objects", "Separate Objects", False),
                AlgorithmProperty("clip", "Clip area", False),
                AlgorithmProperty("units", "Units", UNITS_LIST[2], possible_values=UNITS_LIST, property_type=list)]

    @classmethod
    def save(cls, save_location: typing.Union[str, BytesIO, Path], project_info: ProjectTuple, parameters: dict):
        save_cmap(save_location, project_info.image, project_info.segmentation, parameters)


class SaveXYZ(SaveBase):
    @classmethod
    def get_name(cls):
        return "XYZ text (*.xyz *.txt)"

    @classmethod
    def get_short_name(cls):
        return "xyz"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("channel", "Channel", 0, property_type=Channel),
                AlgorithmProperty("separated_objects", "Separate Objects", False),
                AlgorithmProperty("clip", "Clip area", False)]

    @classmethod
    def _save(cls, save_location, channel_image, segmentation_mask, shift):
        positions = np.transpose(np.nonzero(segmentation_mask))
        positions = np.flip(positions, 1)
        positions -= shift
        values = channel_image[segmentation_mask]
        values = values.reshape(values.size, 1)
        data = np.append(positions, values, axis=1)
        if np.issubdtype(channel_image.dtype, np.integer):
            fm = "%d"
        else:
            fm = "%f"
        # noinspection PyTypeChecker
        np.savetxt(save_location, data, fmt=['%d'] * channel_image.ndim + [fm], delimiter=" ")

    @classmethod
    def save(cls, save_location: typing.Union[str, BytesIO, Path], project_info: ProjectTuple, parameters: dict):
        print(f"[save]", save_location)
        if project_info.segmentation is None:
            raise ValueError("Not segmentation")
        if isinstance(save_location, (str, Path)):
            if not os.path.exists(os.path.dirname(save_location)):
                os.makedirs(os.path.dirname(save_location))
        if parameters.get("separated_objects", False) and not isinstance(save_location, (str, Path)):
            raise ValueError("Saving components to buffer not supported")
        channel_image = project_info.image.get_channel(parameters["channel"])
        segmentation_mask = np.array(project_info.segmentation > 0)
        if parameters.get("clip", False):
            positions = np.transpose(np.nonzero(segmentation_mask))
            positions = np.flip(positions, 1)
            shift = np.min(positions, 0)
        else:
            shift = np.array([0] * segmentation_mask.ndim)
        cls._save(save_location, channel_image, segmentation_mask, shift)
        if parameters.get("separated_objects", False):
            components_count = np.bincount(project_info.segmentation.flat)
            for i, size in enumerate(components_count[1:], 1):
                if size > 0:
                    segmentation_mask = np.array(project_info.segmentation == i)
                    base_path, ext = os.path.splitext(save_location)
                    new_save_location = base_path + f"_part{i}" + ext
                    cls._save(new_save_location, channel_image, segmentation_mask, shift)


save_register.register(SaveProject)
save_register.register(SaveCmap)
save_register.register(SaveXYZ)
