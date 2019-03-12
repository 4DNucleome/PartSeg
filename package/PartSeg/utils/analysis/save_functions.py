from pathlib import Path

import tifffile

from .io_utils import ProjectTuple
from ..analysis.analysis_utils import HistoryElement
from ..channel_class import Channel
from ..algorithm_describe_base import AlgorithmProperty
from ..universal_const import UNIT_SCALE, Units
from ..analysis.save_hooks import PartEncoder
import numpy as np
from PartSeg.tiff_image import Image, ImageWriter
import tarfile
from io import BytesIO
import h5py
import typing
import os.path
import json
from ..io_utils import get_tarinfo, SaveBase
from ..analysis.save_register import save_dict


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


def save_cmap(file: typing.Union[str, h5py.File, BytesIO], image: Image, segmentation: np.ndarray, full_segmentation: np.ndarray,
              cmap_profile: dict, metadata: typing.Optional[dict] = None):
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

    if cmap_profile["reverse"]:
        if full_segmentation is None:
            full_segmentation = segmentation
        mean_val = np.mean(data[full_segmentation == 0])
        data = mean_val - data
        data[data < 0] = 0

    data[segmentation == 0] = 0
    grp = cmap_file.create_group('Chimera/image1')
    if cmap_profile["clip"]:
        points = np.nonzero(segmentation)
        lower_bound = np.min(points, axis=1)
        upper_bound = np.max(points, axis=1)
        cut_img = np.zeros(upper_bound - lower_bound + np.array([6, 6, 6]), dtype=data.dtype)
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
    grp.attrs['step'] = np.array(image._image_spacing, dtype=np.float32)[::-1] * UNIT_SCALE[cmap_profile["units"].value]

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
    def save(cls, save_location: typing.Union[str, BytesIO, Path], project_info: ProjectTuple, parameters: dict,
             range_changed=None, step_changed=None):
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
                AlgorithmProperty("units", "Units", Units.nm, property_type=Units),
                AlgorithmProperty('reverse', 'Reverse', False,
                                  tool_tip="Reverse brightness off image (for electron microscopy)")]

    @classmethod
    def save(cls, save_location: typing.Union[str, BytesIO, Path], project_info: ProjectTuple, parameters: dict,
             range_changed=None, step_changed=None):
        if parameters["separated_objects"] and isinstance(save_location, (str, Path)):
            for i in range(1, project_info.segmentation.max()+1):
                seg = (project_info.segmentation == i).astype(np.uint8)
                if np.any(seg):
                    base, ext = os.path.splitext(save_location)
                    save_loc = base + f"_comp{i}" + ext
                    save_cmap(save_loc, project_info.image, seg,
                              project_info.full_segmentation,
                              parameters)
        else:
            save_cmap(save_location, project_info.image, project_info.segmentation, project_info.full_segmentation,
                      parameters)


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
    def save(cls, save_location: typing.Union[str, BytesIO, Path], project_info: ProjectTuple, parameters: dict,
             range_changed=None, step_changed=None):
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


class SaveAsTiff(SaveBase):
    @classmethod
    def get_name(cls):
        return "Image (*.tiff *.tif)"

    @classmethod
    def get_short_name(cls):
        return "tiff"

    @classmethod
    def get_fields(cls):
        return []

    @classmethod
    def save(cls, save_location: typing.Union[str, BytesIO, Path], project_info, parameters: dict,
             range_changed=None, step_changed=None):
        ImageWriter.save(project_info.image, save_location)


class SaveAsNumpy(SaveBase):
    @classmethod
    def get_name(cls):
        return "Numpy Image (*.npy)"

    @classmethod
    def get_short_name(cls):
        return "numpy_image"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("squeeze", "Squeeze  array", False,
                                  tool_tip="Remove single-dimensional entries from the shape of an array")]

    @classmethod
    def save(cls, save_location: typing.Union[str, BytesIO, Path], project_info, parameters: dict,
             range_changed=None, step_changed=None):
        data = project_info.image.get_data()
        if parameters["squeeze"]:
            data = np.squeeze(data)
        np.save(save_location, data)


class SaveMaskAsTiff(SaveBase):
    @classmethod
    def get_name(cls):
        return "Mask (*.tiff *.tif)"

    @classmethod
    def get_short_name(cls):
        return "mask_tiff"

    @classmethod
    def get_fields(cls):
        return []

    @classmethod
    def save(cls, save_location: typing.Union[str, BytesIO, Path], project_info, parameters: dict,
             range_changed=None, step_changed=None):
        ImageWriter.save_mask(project_info.image, save_location)


class SaveSegmentationAsTIFF(SaveBase):
    @classmethod
    def get_name(cls):
        return "Segmentation (*.tiff *.tif)"

    @classmethod
    def get_short_name(cls):
        return "segmentation_tiff"

    @classmethod
    def get_fields(cls):
        return []

    @classmethod
    def save(cls, save_location: typing.Union[str, BytesIO, Path], project_info, parameters: dict,
             range_changed=None, step_changed=None):
        tifffile.imsave(save_location, project_info.segmentation)

class SaveSegmentationAsNumpy(SaveBase):
    @classmethod
    def get_name(cls):
        return "Segmentation (*.npy)"

    @classmethod
    def get_short_name(cls):
        return "segmentation_numpy"

    @classmethod
    def get_fields(cls):
        return []

    @classmethod
    def save(cls, save_location: typing.Union[str, BytesIO, Path], project_info, parameters: dict,
             range_changed=None, step_changed=None):
        np.save(save_location, project_info.segmentation)


save_dict.register(SaveProject)
save_dict.register(SaveCmap)
save_dict.register(SaveXYZ)
save_dict.register(SaveAsTiff)
save_dict.register(SaveMaskAsTiff)
save_dict.register(SaveAsNumpy)
save_dict.register(SaveSegmentationAsTIFF)
save_dict.register(SaveSegmentationAsNumpy)
