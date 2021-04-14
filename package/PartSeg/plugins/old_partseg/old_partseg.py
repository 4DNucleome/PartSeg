import json
import tarfile
import typing
from io import BufferedIOBase, BytesIO, IOBase, RawIOBase, TextIOBase
from pathlib import Path

import numpy as np

from PartSegCore.analysis.io_utils import ProjectTuple
from PartSegCore.io_utils import LoadBase
from PartSegCore.roi_info import ROIInfo
from PartSegCore.segmentation.noise_filtering import DimensionType
from PartSegCore.universal_const import UNIT_SCALE, Units
from PartSegImage import Image


class LoadPartSegOld(LoadBase):
    @classmethod
    def get_name(cls):
        return "Project old (*.tgz *.tbz2 *.gz *.bz2)"

    @classmethod
    def get_short_name(cls):
        return "project_old"

    # noinspection DuplicatedCode
    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
        metadata: typing.Optional[dict] = None,
    ):
        """Load project from archive old format"""
        file_ob: typing.Union[str, tarfile.TarFile, TextIOBase, BufferedIOBase, RawIOBase, IOBase] = load_locations[0]
        if isinstance(file_ob, tarfile.TarFile):
            tar_file = file_ob
            file_path = ""
        elif isinstance(file_ob, str):
            tar_file = tarfile.open(file_ob)
            file_path = file_ob
        elif isinstance(file_ob, (TextIOBase, BufferedIOBase, RawIOBase, IOBase)):
            tar_file = tarfile.open(fileobj=file_ob)
            file_path = ""
        else:
            raise ValueError(f"wrong type of file_ argument: {type(file_ob)}")
        image_buffer = BytesIO()
        image_tar = tar_file.extractfile(tar_file.getmember("image.npy"))
        image_buffer.write(image_tar.read())
        image_buffer.seek(0)
        image_arr = np.load(image_buffer)
        try:
            res_buffer = BytesIO()
            res_tar = tar_file.extractfile(tar_file.getmember("res_mask.npy"))
            res_buffer.write(res_tar.read())
            res_buffer.seek(0)
            seg_array = np.load(res_buffer)
        except KeyError:
            seg_array = None
        algorithm_str = tar_file.extractfile("data.json").read()
        algorithm_dict = json.loads(algorithm_str)
        spacing = np.array(algorithm_dict["spacing"][::-1]) / UNIT_SCALE[Units.nm.value]
        image = Image(image_arr.reshape((1,) + image_arr.shape + (1,)), spacing, file_path)
        values = {
            "channel": 0,
            "minimum_size": algorithm_dict["minimum_size"],
            "threshold": {"name": "Manual", "values": {"threshold": algorithm_dict["threshold"]}},
            "noise_removal": {"name": "Gauss", "values": {"gauss_type": DimensionType.Layer, "radius": 1.0}}
            if algorithm_dict["use_gauss"]
            else {"name": "None", "values": {}},
            "side_connection": True,
        }

        algorithm_parameters = {
            "name": "Upper threshold" if algorithm_dict["threshold_type"] == "Upper" else "Lower threshold",
            "values": values,
            "threshold_list": algorithm_dict["threshold_list"],
        }

        return ProjectTuple(file_path, image, ROIInfo(seg_array), algorithm_parameters=algorithm_parameters)
