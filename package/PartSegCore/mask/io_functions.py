import os
import tarfile
import typing
from collections import defaultdict
from functools import partial
from pathlib import Path
from io import BytesIO, TextIOBase, BufferedIOBase, RawIOBase, IOBase

import numpy as np
import json

import tifffile

from ..json_hooks import ProfileEncoder
from ..io_utils import (
    get_tarinfo,
    SaveBase,
    LoadBase,
    proxy_callback,
    ProjectInfoBase,
    check_segmentation_type,
    SegmentationType,
    WrongFileTypeException,
    UpdateLoadedMetadataBase,
    open_tar_file,
    HistoryElement,
    SaveMaskAsTiff,
    tar_to_buff,
)
from ..algorithm_describe_base import AlgorithmProperty, Register, SegmentationProfile
from PartSegImage import Image, ImageWriter, GenericImageReader


class SegmentationTuple(ProjectInfoBase, typing.NamedTuple):
    file_path: str
    image: typing.Union[Image, str, None]
    mask: typing.Optional[np.ndarray] = None
    segmentation: typing.Optional[np.ndarray] = None
    selected_components: typing.List = []
    segmentation_parameters: typing.Dict[int, typing.Optional[SegmentationProfile]] = {}
    history: typing.List[HistoryElement] = []
    errors: str = ""

    def get_raw_copy(self):
        return SegmentationTuple(self.file_path, self.image.substitute())

    def is_raw(self):
        return self.segmentation is None

    def replace_(self, *args, **kwargs):
        return self._replace(*args, **kwargs)


def save_stack_segmentation(
    file_data: typing.Union[tarfile.TarFile, str, Path, TextIOBase, BufferedIOBase, RawIOBase, IOBase],
    segmentation_info: SegmentationTuple,
    parameters: dict,
    range_changed=None,
    step_changed=None,
):
    if range_changed is None:
        range_changed = empty_fun
    if step_changed is None:
        step_changed = empty_fun
    range_changed(0, 7)
    tar_file, file_path = open_tar_file(file_data, "w")
    step_changed(1)
    try:
        segmentation_buff = BytesIO()
        # noinspection PyTypeChecker
        tifffile.imwrite(segmentation_buff, segmentation_info.segmentation, compress=9)
        segmentation_tar = get_tarinfo("segmentation.tif", segmentation_buff)
        tar_file.addfile(segmentation_tar, fileobj=segmentation_buff)
        step_changed(3)
        metadata = {
            "components": [int(x) for x in segmentation_info.selected_components],
            "parameters": {str(k): v for k, v in segmentation_info.segmentation_parameters.items()},
            "shape": segmentation_info.segmentation.shape,
        }
        if isinstance(segmentation_info.image, Image):
            file_path = segmentation_info.image.file_path
        elif isinstance(segmentation_info.image, str):
            file_path = segmentation_info.image
        else:
            file_path = ""
        if file_path != "":
            if parameters["relative_path"] and isinstance(file_data, str):
                metadata["base_file"] = os.path.relpath(file_path, os.path.dirname(file_data))
            else:
                metadata["base_file"] = file_path
        metadata_buff = BytesIO(json.dumps(metadata, cls=ProfileEncoder).encode("utf-8"))
        metadata_tar = get_tarinfo("metadata.json", metadata_buff)
        tar_file.addfile(metadata_tar, metadata_buff)
        step_changed(4)
        if segmentation_info.mask is not None:
            mask = segmentation_info.mask
            if mask.dtype == np.bool:
                mask = mask.astype(np.uint8)
            mask_buff = BytesIO()
            tifffile.imwrite(mask_buff, mask, compress=9)
            mask_tar = get_tarinfo("mask.tif", mask_buff)
            tar_file.addfile(mask_tar, fileobj=mask_buff)
        step_changed(5)
        el_info = []
        for i, hist in enumerate(segmentation_info.history):
            el_info.append(
                {
                    "index": i,
                    "mask_property": hist.mask_property,
                    "segmentation_parameters": hist.segmentation_parameters,
                }
            )
            hist.arrays.seek(0)
            hist_info = get_tarinfo(f"history/arrays_{i}.npz", hist.arrays)
            hist.arrays.seek(0)
            tar_file.addfile(hist_info, hist.arrays)
        if len(el_info) > 0:
            hist_str = json.dumps(el_info, cls=ProfileEncoder)
            hist_buff = BytesIO(hist_str.encode("utf-8"))
            tar_algorithm = get_tarinfo("history/history.json", hist_buff)
            tar_file.addfile(tar_algorithm, hist_buff)
        step_changed(6)

    finally:
        if isinstance(file_data, (str, Path)):
            tar_file.close()
    step_changed(6)


def load_stack_segmentation(file_data: typing.Union[str, Path], range_changed=None, step_changed=None):
    if range_changed is None:
        range_changed = empty_fun
    if step_changed is None:
        step_changed = empty_fun
    range_changed(0, 7)
    tar_file = open_tar_file(file_data)[0]
    try:
        if check_segmentation_type(tar_file) != SegmentationType.mask:
            raise WrongFileTypeException()
        files = tar_file.getnames()
        step_changed(1)
        metadata = load_metadata(tar_file.extractfile("metadata.json").read().decode("utf8"))
        step_changed(2)
        if "segmentation.npy" in files:
            segmentation_file_name = "segmentation.npy"
            segmentation_load_fun = np.load
        else:
            segmentation_file_name = "segmentation.tif"
            segmentation_load_fun = tifffile.imread
        segmentation_buff = BytesIO()
        segmentation_tar = tar_file.extractfile(tar_file.getmember(segmentation_file_name))
        segmentation_buff.write(segmentation_tar.read())
        step_changed(3)
        segmentation_buff.seek(0)
        segmentation = segmentation_load_fun(segmentation_buff)
        step_changed(4)
        if "mask.tif" in tar_file.getnames():
            mask = tifffile.imread(tar_to_buff(tar_file, "mask.tif"))
            if np.max(mask) == 1:
                mask = mask.astype(np.bool)
        else:
            mask = None
        step_changed(5)
        history = []
        try:
            history_buff = tar_file.extractfile(tar_file.getmember("history/history.json")).read()
            history_json = load_metadata(history_buff)
            for el in history_json:
                history_buffer = BytesIO()
                history_buffer.write(tar_file.extractfile(f"history/arrays_{el['index']}.npz").read())
                history_buffer.seek(0)
                history.append(
                    HistoryElement(
                        segmentation_parameters=el["segmentation_parameters"],
                        mask_property=el["mask_property"],
                        arrays=history_buffer,
                    )
                )

        except KeyError:
            pass
        step_changed(6)
    finally:
        if isinstance(file_data, (str, Path)):
            tar_file.close()
    return SegmentationTuple(
        file_path=file_data if isinstance(file_data, str) else "",
        image=metadata["base_file"] if "base_file" in metadata else None,
        segmentation=segmentation,
        selected_components=metadata["components"],
        mask=mask,
        segmentation_parameters=metadata["parameters"] if "parameters" in metadata else None,
        history=history,
    )


def empty_fun(_a0=None, _a1=None):
    pass


class LoadSegmentation(LoadBase):
    @classmethod
    def get_name(cls):
        return "Segmentation to image (*.seg *.tgz)"

    @classmethod
    def get_short_name(cls):
        return "seg"

    @staticmethod
    def fix_parameters(profile: SegmentationProfile):
        if profile is None:
            return
        if profile.algorithm == "Threshold" or profile.algorithm == "Auto Threshold":
            if isinstance(profile.values["smooth_border"], bool):
                if profile.values["smooth_border"] and "smooth_border_radius" in profile.values:
                    profile.values["smooth_border"] = {
                        "name": "Opening",
                        "values": {"smooth_border_radius": profile.values["smooth_border_radius"]},
                    }
                    del profile.values["smooth_border_radius"]
                else:
                    profile.values["smooth_border"] = {"name": "None", "values": {}}
        return profile

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
        metadata: typing.Optional[dict] = None,
    ) -> SegmentationTuple:
        segmentation_tuple = load_stack_segmentation(
            load_locations[0], range_changed=range_changed, step_changed=step_changed
        )
        if segmentation_tuple.segmentation_parameters is None:
            parameters = defaultdict(lambda: None)
        else:
            parameters = defaultdict(
                lambda: None,
                [(int(k), cls.fix_parameters(v)) for k, v in segmentation_tuple.segmentation_parameters.items()],
            )
        return segmentation_tuple.replace_(segmentation_parameters=parameters)

    @classmethod
    def partial(cls):
        return False


class LoadSegmentationParameters(LoadBase):
    @classmethod
    def get_name(cls):
        return "Segmentation parameters (*.json *.seg *.tgz)"

    @classmethod
    def get_short_name(cls):
        return "seg_par"

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
        metadata: typing.Optional[dict] = None,
    ) -> SegmentationTuple:
        file_data = load_locations[0]

        if isinstance(file_data, (str, Path)):
            ext = os.path.splitext(file_data)[1]
            if ext == ".json":
                metadata = load_metadata(file_data)
                if isinstance(metadata, SegmentationProfile):
                    parameters = {1: metadata}
                else:
                    parameters = defaultdict(
                        lambda: None,
                        [(int(k), LoadSegmentation.fix_parameters(v)) for k, v in metadata["parameters"].items()],
                    )
                return SegmentationTuple(file_path=file_data, image=None, segmentation_parameters=parameters)

        tar_file, _ = open_tar_file(file_data)
        try:
            metadata = load_metadata(tar_file.extractfile("metadata.json").read().decode("utf8"))
            parameters = defaultdict(
                lambda: None, [(int(k), LoadSegmentation.fix_parameters(v)) for k, v in metadata["parameters"].items()]
            )
        finally:
            if isinstance(file_data, (str, Path)):
                tar_file.close()
        return SegmentationTuple(file_path=file_data, image=None, segmentation_parameters=parameters)


class LoadSegmentationImage(LoadBase):
    @classmethod
    def get_name(cls):
        return "Segmentation with image (*.seg *.tgz)"

    @classmethod
    def get_short_name(cls):
        return "seg_img"

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
        metadata: typing.Optional[dict] = None,
    ) -> SegmentationTuple:
        seg = LoadSegmentation.load(load_locations)
        if len(load_locations) > 1:
            base_file = load_locations[1]
        else:
            base_file = seg.image
        if base_file is None:
            raise IOError(f"base file for segmentation not defined")
        if os.path.isabs(base_file):
            file_path = base_file
        else:
            if not isinstance(load_locations[0], str):
                raise IOError(f"Cannot use relative path {base_file} for non path argument")
            file_path = os.path.join(os.path.dirname(load_locations[0]), base_file)
        if not os.path.exists(file_path):
            raise IOError(f"Base file for segmentation do not exists: {base_file} -> {file_path}")
        if metadata is None:
            metadata = {"default_spacing": (10 ** -6, 10 ** -6, 10 ** -6)}
        image = GenericImageReader.read_image(
            file_path,
            callback_function=partial(proxy_callback, range_changed, step_changed),
            default_spacing=metadata["default_spacing"],
        )
        # noinspection PyProtectedMember
        # image.file_path = load_locations[0]
        return seg.replace_(file_path=image.file_path, image=image)


class LoadStackImage(LoadBase):
    @classmethod
    def get_name(cls):
        return "Image(*.tif *.tiff *.lsm *.czi *.oib *.oif)"

    @classmethod
    def get_short_name(cls):
        return "img"

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
        metadata: typing.Optional[dict] = None,
    ) -> SegmentationTuple:
        if metadata is None:
            metadata = {"default_spacing": (10 ** -6, 10 ** -6, 10 ** -6)}
        image = GenericImageReader.read_image(
            load_locations[0],
            callback_function=partial(proxy_callback, range_changed, step_changed),
            default_spacing=metadata["default_spacing"],
        )

        return SegmentationTuple(image.file_path, image, selected_components=[])


class LoadStackImageWithMask(LoadBase):
    @classmethod
    def get_short_name(cls):
        return "img_with_mask"

    @classmethod
    def get_next_file(cls, file_paths: typing.List[str]):
        base, ext = os.path.splitext(file_paths[0])
        return base + "_mask" + ext

    @classmethod
    def number_of_files(cls):
        return 2

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
        metadata: typing.Optional[dict] = None,
    ) -> typing.Union[ProjectInfoBase, typing.List[ProjectInfoBase]]:
        if metadata is None:
            metadata = {"default_spacing": (10 ** -6, 10 ** -6, 10 ** -6)}
        image = GenericImageReader.read_image(
            load_locations[0],
            load_locations[1],
            callback_function=partial(proxy_callback, range_changed, step_changed),
            default_spacing=metadata["default_spacing"],
        )

        return SegmentationTuple(image.file_path, image, mask=image.mask, selected_components=[])

    @classmethod
    def get_name(cls) -> str:
        return "Image with mask(*.tif *.tiff *.lsm *.czi *.oib *.oif)"


class SaveSegmentation(SaveBase):
    @classmethod
    def get_name(cls):
        return "Segmentation (*.seg *.tgz)"

    @classmethod
    def get_short_name(cls):
        return "seg"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("relative_path", "Relative Path\nin segmentation", False)]

    @classmethod
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info: SegmentationTuple,
        parameters: dict,
        range_changed=None,
        step_changed=None,
    ):
        save_stack_segmentation(save_location, project_info, parameters)


def save_components(
    image: Image, components: list, segmentation: np.ndarray, dir_path: str, range_changed=None, step_changed=None
):
    if range_changed is None:
        range_changed = empty_fun
    if step_changed is None:
        step_changed = empty_fun

    file_name = os.path.splitext(os.path.basename(image.file_path))[0]
    range_changed(0, 2 * len(components))
    for i in components:
        im = image.cut_image(segmentation == i, replace_mask=True)
        # print(f"[run] {im}")
        ImageWriter.save(im, os.path.join(dir_path, f"{file_name}_component{i}.tif"))
        step_changed(2 * i + 1)
        ImageWriter.save_mask(im, os.path.join(dir_path, f"{file_name}_component{i}_mask.tif"))
        step_changed(2 * i + 2)


class SaveComponents(SaveBase):
    @classmethod
    def get_short_name(cls):
        return "comp"

    @classmethod
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info: SegmentationTuple,
        parameters: dict,
        range_changed=None,
        step_changed=None,
    ):
        save_components(
            project_info.image,
            project_info.selected_components,
            project_info.segmentation,
            save_location,
            range_changed,
            step_changed,
        )

    @classmethod
    def get_name(cls) -> str:
        return "Components"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return []


class SaveParametersJSON(SaveBase):
    @classmethod
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info,
        parameters: dict = None,
        range_changed=None,
        step_changed=None,
    ):
        """
        :param save_location: path to save
        :param project_info: data to save in json file
        :param parameters: Not used, keep for satisfy interface
        :param range_changed: Not used, keep for satisfy interface
        :param step_changed: Not used, keep for satisfy interface
        :return:
        """
        with open(save_location, "w") as ff:
            json.dump(project_info, ff, cls=ProfileEncoder)

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return []

    @classmethod
    def get_short_name(cls):
        return "json"

    @classmethod
    def get_name(cls) -> str:
        return "Parameters (*.json)"


def load_metadata(data: typing.Union[str, Path, typing.TextIO]):
    """
    Load metadata saved in json format for segmentation mask
    :param data: path to json file, string with json, or opened file
    :return: restored structures
    """
    return UpdateLoadedMetadataMask.load_json_data(data)


class UpdateLoadedMetadataMask(UpdateLoadedMetadataBase):
    @classmethod
    def update_segmentation_profile(cls, profile_data: SegmentationProfile) -> SegmentationProfile:
        profile_data = super().update_segmentation_profile(profile_data)
        if profile_data.algorithm == "Threshold" or profile_data.algorithm == "Auto Threshold":
            if isinstance(profile_data.values["smooth_border"], bool):
                if profile_data.values["smooth_border"]:
                    profile_data.values["smooth_border"] = {
                        "name": "Opening",
                        "values": {"smooth_border_radius": profile_data.values["smooth_border_radius"]},
                    }
                else:
                    profile_data.values["smooth_border"] = {"name": "None", "values": {}}
                if "smooth_border_radius" in profile_data.values:
                    del profile_data.values["smooth_border_radius"]
            if "noise_removal" in profile_data.values:
                profile_data.values["noise_filtering"] = profile_data.values["noise_removal"]
                del profile_data.values["noise_removal"]
        return profile_data


load_dict = Register(
    LoadStackImage, LoadSegmentationImage, LoadStackImageWithMask, class_methods=LoadBase.need_functions
)
save_parameters_dict = Register(SaveParametersJSON, class_methods=SaveBase.need_functions)
save_components_dict = Register(SaveComponents, class_methods=SaveBase.need_functions)
save_segmentation_dict = Register(SaveSegmentation, SaveMaskAsTiff, class_methods=SaveBase.need_functions)
