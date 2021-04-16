import json
import os.path
import tarfile
import typing
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import tifffile

from PartSegImage import Image, ImageWriter

from ..algorithm_describe_base import AlgorithmProperty, Register
from ..channel_class import Channel
from ..io_utils import NotSupportedImage, SaveBase, SaveMaskAsTiff, SaveROIAsNumpy, SaveROIAsTIFF, get_tarinfo
from ..project_info import HistoryElement
from ..roi_info import ROIInfo
from ..universal_const import UNIT_SCALE, Units
from .io_utils import ProjectTuple, project_version_info
from .save_hooks import PartEncoder

__all__ = [
    "SaveProject",
    "SaveCmap",
    "SaveXYZ",
    "SaveAsTiff",
    "SaveAsNumpy",
    "save_dict",
]


# TODO add progress function to io
def save_project(
    file_path: str,
    image: Image,
    roi_info: ROIInfo,
    mask: typing.Optional[np.ndarray],
    history: typing.List[HistoryElement],
    algorithm_parameters: dict,
):
    # TODO add support for binary objects
    ext = os.path.splitext(file_path)[1]
    tar_mod = "w:bz2" if ext.lower() in [".bz2", ".tbz2"] else "w:gz"
    with tarfile.open(file_path, tar_mod) as tar:
        segmentation_buff = BytesIO()
        # noinspection PyTypeChecker
        tifffile.imwrite(segmentation_buff, roi_info.roi, compress=9)
        segmentation_tar = get_tarinfo("segmentation.tif", segmentation_buff)
        tar.addfile(segmentation_tar, fileobj=segmentation_buff)
        if roi_info.alternative:
            alternative_buff = BytesIO()
            np.savez(alternative_buff, **roi_info.alternative)
            alternative_tar = get_tarinfo("alternative.npz", alternative_buff)
            tar.addfile(alternative_tar, fileobj=alternative_buff)
        if mask is not None:
            if mask.dtype == bool:
                mask = mask.astype(np.uint8)
            segmentation_buff = BytesIO()
            # noinspection PyTypeChecker
            tifffile.imwrite(segmentation_buff, mask, compress=9)
            segmentation_tar = get_tarinfo("mask.tif", segmentation_buff)
            tar.addfile(segmentation_tar, fileobj=segmentation_buff)
        image_buff = BytesIO()
        ImageWriter.save(image, image_buff)
        tar_image = get_tarinfo("image.tif", image_buff)
        tar.addfile(tarinfo=tar_image, fileobj=image_buff)
        para_str = json.dumps(algorithm_parameters, cls=PartEncoder)
        parameters_buff = BytesIO(para_str.encode("utf-8"))
        tar_algorithm = get_tarinfo("algorithm.json", parameters_buff)
        tar.addfile(tar_algorithm, parameters_buff)
        meta_str = json.dumps(
            {"project_version_info": str(project_version_info), "roi_annotations": roi_info.annotations},
            cls=PartEncoder,
        )
        meta_buff = BytesIO(meta_str.encode("utf-8"))
        tar_meta = get_tarinfo("metadata.json", meta_buff)
        tar.addfile(tar_meta, meta_buff)
        el_info = []
        for i, el in enumerate(history):
            el_info.append(
                {
                    "index": i,
                    "algorithm_name": el.roi_extraction_parameters["algorithm_name"],
                    "values": el.roi_extraction_parameters["values"],
                    "mask_property": el.mask_property,
                    "annotations": el.annotations,
                }
            )
            el.arrays.seek(0)
            hist_info = get_tarinfo(f"history/arrays_{i}.npz", el.arrays)
            el.arrays.seek(0)
            tar.addfile(hist_info, el.arrays)
        if el_info:
            hist_str = json.dumps(el_info, cls=PartEncoder)
            hist_buff = BytesIO(hist_str.encode("utf-8"))
            tar_algorithm = get_tarinfo("history/history.json", hist_buff)
            tar.addfile(tar_algorithm, hist_buff)


def save_cmap(
    file: typing.Union[str, h5py.File, BytesIO],
    data: np.ndarray,
    spacing,
    segmentation: np.ndarray,
    reverse_base: float,
    cmap_profile: dict,
    metadata: typing.Optional[dict] = None,
):
    if segmentation is None or np.max(segmentation) == 0:
        raise ValueError("No segmentation")
    if isinstance(file, (str, BytesIO, Path)):
        if isinstance(file, str) and os.path.exists(file):
            os.remove(file)
        cmap_file = h5py.File(file, "w")
    elif isinstance(file, h5py.File):
        cmap_file = file
    else:
        raise ValueError(f"Wrong type of file argument, type: {type(file)}")
    if cmap_profile["reverse"]:
        data = reverse_base - data
        data[data < 0] = 0
    data = np.copy(data)
    data[segmentation == 0] = 0
    grp = cmap_file.create_group("Chimera/image1")
    z, y, x = data.shape
    data_set = grp.create_dataset("data_zyx", (z, y, x), dtype="f", compression="gzip")
    data_set[...] = data

    if metadata:
        meta_group = cmap_file.create_group("Chimera/image1/Statistics")
        for key, val in metadata.items():
            meta_group.attrs[key] = val

    grp = cmap_file["Chimera"]
    grp.attrs["CLASS"] = np.string_("GROUP")
    grp.attrs["TITLE"] = np.string_("")
    grp.attrs["VERSION"] = np.string_("1.0")

    grp = cmap_file["Chimera/image1"]
    grp.attrs["CLASS"] = np.string_("GROUP")
    grp.attrs["TITLE"] = np.string_("")
    grp.attrs["VERSION"] = np.string_("1.0")
    grp.attrs["step"] = np.array(spacing, dtype=np.float32)[::-1] * UNIT_SCALE[cmap_profile["units"].value]

    if isinstance(file, str):
        cmap_file.close()


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
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info: ProjectTuple,
        parameters: dict = None,
        range_changed=None,
        step_changed=None,
    ):
        save_project(
            save_location,
            project_info.image,
            project_info.roi_info,
            project_info.mask,
            project_info.history,
            project_info.algorithm_parameters,
        )


class SaveCmap(SaveBase):
    @classmethod
    def get_name(cls):
        return "Chimera CMAP (*.cmap)"

    @classmethod
    def get_short_name(cls):
        return "cmap"

    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty("channel", "Channel", 0, value_type=Channel),
            AlgorithmProperty("separated_objects", "Separate Objects", False),
            AlgorithmProperty("clip", "Clip area", False),
            AlgorithmProperty("units", "Units", Units.nm, value_type=Units),
            AlgorithmProperty(
                "reverse", "Reverse", False, help_text="Reverse brightness off image (for electron microscopy)"
            ),
        ]

    @classmethod
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info: ProjectTuple,
        parameters: dict,
        range_changed=None,
        step_changed=None,
    ):

        if project_info.image.shape[project_info.image.time_pos] != 1:
            raise NotSupportedImage("This save method o not support time data")
        data = project_info.image.get_data_by_axis(c=parameters["channel"], t=0)
        spacing = project_info.image.spacing
        segmentation = project_info.image.clip_array(project_info.roi_info.roi, t=0)

        reverse_base = float(np.mean(data[segmentation == 0]))
        if parameters.get("clip", False):
            positions = np.transpose(np.nonzero(segmentation))
            clip_down = np.min(positions, 0)
            clip_up = np.max(positions, 0)
            clip = tuple(slice(x, y + 1) for x, y in zip(clip_down, clip_up))
            data = data[clip]
            segmentation = segmentation[clip]

        if parameters["separated_objects"] and isinstance(save_location, (str, Path)):
            for i in range(1, np.max(segmentation) + 1):
                seg = (segmentation == i).astype(np.uint8)
                if np.any(seg):
                    base, ext = os.path.splitext(save_location)
                    save_loc = base + f"_comp{i}" + ext
                    save_cmap(save_loc, data, spacing, seg, reverse_base, parameters)
        else:
            save_cmap(save_location, data, spacing, segmentation, reverse_base, parameters)


class SaveXYZ(SaveBase):
    @classmethod
    def get_name(cls):
        return "XYZ text (*.xyz *.txt)"

    @classmethod
    def get_short_name(cls):
        return "xyz"

    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty("channel", "Channel", 0, value_type=Channel),
            AlgorithmProperty("separated_objects", "Separate Objects", False),
            AlgorithmProperty("clip", "Clip area", False),
        ]

    @classmethod
    def _save(cls, save_location, channel_image, segmentation_mask, shift):
        positions = np.transpose(np.nonzero(segmentation_mask))
        positions = np.flip(positions, 1)
        positions -= shift
        values = channel_image[segmentation_mask]
        values = values.reshape(values.size, 1)
        data = np.append(positions, values, axis=1)
        fm = "%d" if np.issubdtype(channel_image.dtype, np.integer) else "%f"
        # noinspection PyTypeChecker
        np.savetxt(save_location, data, fmt=["%d"] * channel_image.ndim + [fm], delimiter=" ")

    @classmethod
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info: ProjectTuple,
        parameters: dict,
        range_changed=None,
        step_changed=None,
    ):
        if project_info.roi_info.roi is None:
            raise ValueError("Not segmentation")
        if isinstance(save_location, (str, Path)) and not os.path.exists(os.path.dirname(save_location)):
            os.makedirs(os.path.dirname(save_location))
        if parameters.get("separated_objects", False) and not isinstance(save_location, (str, Path)):
            raise ValueError("Saving components to buffer not supported")
        if project_info.image.shape[project_info.image.time_pos] != 1 and "time" not in parameters:
            raise NotSupportedImage("This save method o not support time data")
        channel_image = project_info.image.get_data_by_axis(c=parameters["channel"], t=parameters.get("time", 0))

        segmentation_mask = np.array(project_info.roi_info.roi > 0)
        segmentation_mask = project_info.image.clip_array(segmentation_mask, t=parameters.get("time", 0))
        if parameters.get("clip", False):
            positions = np.transpose(np.nonzero(segmentation_mask))
            positions = np.flip(positions, 1)
            shift = np.min(positions, 0)
        else:
            shift = np.array([0] * segmentation_mask.ndim)
        cls._save(save_location, channel_image, segmentation_mask, shift)
        if parameters.get("separated_objects", False):
            components_count = np.bincount(project_info.roi_info.roi.flat)
            for i, size in enumerate(components_count[1:], 1):
                if size > 0:
                    segmentation_mask = np.array(project_info.roi_info.roi == i)[parameters.get("time", 0)]
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
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info,
        parameters: dict = None,
        range_changed=None,
        step_changed=None,
    ):
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
        return [
            AlgorithmProperty(
                "squeeze",
                "Squeeze  array",
                False,
                help_text="Remove single-dimensional entries from the shape of an array",
            )
        ]

    @classmethod
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info,
        parameters: dict,
        range_changed=None,
        step_changed=None,
    ):
        data = project_info.image.get_data()
        if parameters["squeeze"]:
            data = np.squeeze(data)
        np.save(save_location, data)


save_dict = Register(
    SaveProject,
    SaveCmap,
    SaveXYZ,
    SaveAsTiff,
    SaveMaskAsTiff,
    SaveAsNumpy,
    SaveROIAsTIFF,
    SaveROIAsNumpy,
    class_methods=SaveBase.need_functions,
)
