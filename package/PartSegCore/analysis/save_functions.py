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
from .io_utils import ProjectTuple, project_version_info
from ..algorithm_describe_base import AlgorithmProperty, Register
from .save_hooks import PartEncoder
from ..channel_class import Channel
from ..io_utils import get_tarinfo, SaveBase, NotSupportedImage, HistoryElement, SaveMaskAsTiff
from ..universal_const import UNIT_SCALE, Units

__all__ = [
    "SaveProject",
    "SaveCmap",
    "SaveXYZ",
    "SaveAsTiff",
    "SaveAsNumpy",
    "SaveSegmentationAsTIFF",
    "SaveSegmentationAsNumpy",
    "save_dict",
]


# TODO add progress function to io
def save_project(
    file_path: str,
    image: Image,
    segmentation: np.ndarray,
    full_segmentation: np.ndarray,
    mask: typing.Optional[np.ndarray],
    history: typing.List[HistoryElement],
    algorithm_parameters: dict,
):
    # TODO add support for binary objects
    ext = os.path.splitext(file_path)[1]
    if ext.lower() in [".bz2", ".tbz2"]:
        tar_mod = "w:bz2"
    else:
        tar_mod = "w:gz"
    with tarfile.open(file_path, tar_mod) as tar:
        segmentation_buff = BytesIO()
        # noinspection PyTypeChecker
        tifffile.imwrite(segmentation_buff, segmentation, compress=9)
        segmentation_tar = get_tarinfo("segmentation.tif", segmentation_buff)
        tar.addfile(segmentation_tar, fileobj=segmentation_buff)
        segmentation_buff = BytesIO()
        # noinspection PyTypeChecker
        tifffile.imwrite(segmentation_buff, full_segmentation, compress=9)
        segmentation_tar = get_tarinfo("full_segmentation.tif", segmentation_buff)
        tar.addfile(segmentation_tar, fileobj=segmentation_buff)
        if mask is not None:
            if mask.dtype == np.bool:
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
        meta_str = json.dumps({"project_version_info": str(project_version_info)}, cls=PartEncoder)
        meta_buff = BytesIO(meta_str.encode("utf-8"))
        tar_meta = get_tarinfo("metadata.json", meta_buff)
        tar.addfile(tar_meta, meta_buff)
        el_info = []
        for i, el in enumerate(history):
            el_info.append(
                {
                    "index": i,
                    "algorithm_name": el.segmentation_parameters["algorithm_name"],
                    "values": el.segmentation_parameters["values"],
                    "mask_property": el.mask_property,
                }
            )
            el.arrays.seek(0)
            hist_info = get_tarinfo(f"history/arrays_{i}.npz", el.arrays)
            el.arrays.seek(0)
            tar.addfile(hist_info, el.arrays)
        if len(el_info) > 0:
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
    if isinstance(file, (str, BytesIO)):
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
            project_info.segmentation,
            project_info.full_segmentation,
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
            AlgorithmProperty("channel", "Channel", 0, property_type=Channel),
            AlgorithmProperty("separated_objects", "Separate Objects", False),
            AlgorithmProperty("clip", "Clip area", False),
            AlgorithmProperty("units", "Units", Units.nm, property_type=Units),
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
        data = project_info.image.get_channel(parameters["channel"])
        if data.shape[0] != 1:
            raise NotSupportedImage("This save method o not support time data")
        data = data[0]
        spacing = project_info.image.spacing
        segmentation = project_info.segmentation
        full_segmentation = project_info.full_segmentation
        if full_segmentation is None:
            full_segmentation = segmentation
        reverse_base = float(np.mean(data[full_segmentation == 0]))
        if parameters.get("clip", False):
            positions = np.transpose(np.nonzero(segmentation))
            clip_down = np.min(positions, 0)
            clip_up = np.max(positions, 0)
            clip = tuple([slice(x, y + 1) for x, y in zip(clip_down, clip_up)])
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
            AlgorithmProperty("channel", "Channel", 0, property_type=Channel),
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
        if np.issubdtype(channel_image.dtype, np.integer):
            fm = "%d"
        else:
            fm = "%f"
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
        if project_info.segmentation is None:
            raise ValueError("Not segmentation")
        if isinstance(save_location, (str, Path)):
            if not os.path.exists(os.path.dirname(save_location)):
                os.makedirs(os.path.dirname(save_location))
        if parameters.get("separated_objects", False) and not isinstance(save_location, (str, Path)):
            raise ValueError("Saving components to buffer not supported")
        channel_image = project_info.image.get_channel(parameters["channel"])
        if channel_image.shape[0] != 1:
            raise NotSupportedImage("This save method o not support time data")
        channel_image = channel_image[0]
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
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info,
        parameters: dict,
        range_changed=None,
        step_changed=None,
    ):
        segmentation = project_info.segmentation
        segmentation_max = segmentation.max()
        if segmentation_max < 2 ** 8 - 1:
            segmentation = segmentation.astype(np.uint8)
        elif segmentation_max < 2 ** 16 - 1:
            segmentation = segmentation.astype(np.uint16)
        tifffile.imsave(save_location, segmentation)


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
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info,
        parameters: dict = None,
        range_changed=None,
        step_changed=None,
    ):
        segmentation = project_info.segmentation
        segmentation_max = segmentation.max()
        if segmentation_max < 2 ** 8 - 1:
            segmentation = segmentation.astype(np.uint8)
        elif segmentation_max < 2 ** 16 - 1:
            segmentation = segmentation.astype(np.uint16)
        np.save(save_location, segmentation)


save_dict = Register(
    SaveProject,
    SaveCmap,
    SaveXYZ,
    SaveAsTiff,
    SaveMaskAsTiff,
    SaveAsNumpy,
    SaveSegmentationAsTIFF,
    SaveSegmentationAsNumpy,
    class_methods=SaveBase.need_functions,
)
