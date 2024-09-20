import inspect
import os.path
import typing
from abc import abstractmethod
from contextlib import suppress
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from threading import Lock

import imagecodecs
import numpy as np
import tifffile
from czifile.czifile import DECOMPRESS, CziFile
from defusedxml import ElementTree
from oiffile import OifFile
from packaging.version import parse as parse_version

from PartSegImage.image import Image

INCOMPATIBLE_IMAGE_MASK = "Incompatible shape of mask and image"


if typing.TYPE_CHECKING:
    from xml.etree.ElementTree import Element  # nosec


CZI_MAX_WORKERS = None


class ZSTD1Header(typing.NamedTuple):
    """
    ZSTD1 header structure
    based on:
    https://github.com/ZEISS/libczi/blob/4a60e22200cbf0c8ff2a59f69a81ef1b2b89bf4f/Src/libCZI/decoder_zstd.cpp#L19
    """

    header_size: int
    hiLoByteUnpackPreprocessing: bool


def parse_zstd1_header(data: bytes, size: int) -> ZSTD1Header:  # pragma: no cover
    """
    Parse ZSTD header

    https://github.com/ZEISS/libczi/blob/4a60e22200cbf0c8ff2a59f69a81ef1b2b89bf4f/Src/libCZI/decoder_zstd.cpp#L84
    """
    if size < 1:
        return ZSTD1Header(0, False)

    if data[0] == 1:
        return ZSTD1Header(1, False)

    if data[0] == 3 and size < 3:
        return ZSTD1Header(0, False)

    if data[1] == 1:
        return ZSTD1Header(3, bool(data[2] & 1))

    return ZSTD1Header(0, False)


def _get_dtype():
    return inspect.currentframe().f_back.f_back.f_locals["de"].dtype


def decode_zstd1(data: bytes) -> np.ndarray:
    """
    Decode ZSTD1 data
    """
    header = parse_zstd1_header(data, len(data))
    dtype = _get_dtype()
    if header.hiLoByteUnpackPreprocessing:
        array_ = np.frombuffer(imagecodecs.zstd_decode(data[header.header_size :]), np.uint8).copy()
        half_size = array_.size // 2
        array = np.empty(half_size, np.uint16)
        array[:] = array_[:half_size] + (array_[half_size:].astype(np.uint16) << 8)
        array = array.view(dtype)
    else:
        array = np.frombuffer(imagecodecs.zstd_decode(data[header.header_size :]), dtype).copy()
    return array


def decode_zstd0(data: bytes) -> np.ndarray:
    """
    Decode ZSTD0 data
    """
    dtype = _get_dtype()
    return np.frombuffer(imagecodecs.zstd_decode(data), dtype).copy()


if parse_version(version("czifile")) == parse_version("2019.7.2"):
    DECOMPRESS[5] = decode_zstd0
    DECOMPRESS[6] = decode_zstd1


def _empty(_, __):
    """Empty function for callback"""


class TiffFileException(Exception):
    """
    exception raised if reading tiff file fails. Created for distinguish exceptions which should
    reported as warning message (not for report)
    """


class BaseImageReader:
    """
    Base class for reading image using Christopher Gholike libraries

    :cvar typing.Type[Image] ~.image_class: image class to return
    """

    image_class = Image

    @classmethod
    def return_order(cls) -> str:
        """
        Order to which image axes should be rearranged before pass to :py:attr:`image_class` constructor.
        Default is :py:attr:`image_class.return_order`
        """
        return cls.image_class.axis_order

    def __init__(self, callback_function=None):
        self.default_spacing = 10**-6, 10**-6, 10**-6
        self.spacing = self.default_spacing
        self.channel_names = None
        if callback_function is None:
            self.callback_function = _empty
        else:
            self.callback_function = callback_function

    def set_default_spacing(self, spacing):
        spacing = tuple(spacing)
        if len(spacing) == 2:
            # one micrometer
            spacing = (10 ** (-6), *spacing)
        if len(spacing) != 3:
            raise ValueError(f"wrong spacing {spacing}")  # pragma: no cover
        self.default_spacing = spacing

    @abstractmethod
    def read(self, image_path: typing.Union[str, Path], mask_path=None, ext=None) -> Image:
        """
        Main function to read image. If ext is not set then it may be deduced from path to file.
        If BytesIO is given and non default data file type is needed then ext need to be set

        :param image_path: path to image or buffer
        :param mask_path: path to mask or buffer
        :param ext: extension to decide algorithm, if absent and image_path is path then
            should be deduced from path
        :return: image structure
        """
        raise NotImplementedError

    @classmethod
    def read_image(
        cls,
        image_path: typing.Union[str, Path],
        mask_path=None,
        callback_function: typing.Optional[typing.Callable] = None,
        default_spacing: typing.Optional[typing.Tuple[float, float, float]] = None,
    ) -> Image:
        """
        read image file with optional mask file

        :param image_path: path or opened file contains image
        :param mask_path:
        :param callback_function: function for provide information about progress in reading file (for progressbar)
        :param default_spacing: used if file do not contains information about spacing
            (or metadata format is not supported)
        :return: image
        """
        # TODO add generic description of callback function
        instance = cls(callback_function)
        if default_spacing is not None:
            instance.set_default_spacing(default_spacing)
        return instance.read(image_path, mask_path)

    @staticmethod
    def _reduce_obsolete_dummy_axes(array, axes) -> typing.Tuple[np.ndarray, str]:
        """
        If there are duplicates in axes string then remove dimensions of size one

        :return: reduced array and axes
        """
        if len(axes) == len(set(axes)):
            return array, axes

        ax_li = []
        shape_li = []
        for dim, ax in zip(array.shape, axes):
            if dim != 1:
                ax_li.append(ax)
                shape_li.append(dim)
        axes = "".join(ax_li)
        array = np.reshape(array, shape_li)
        return array, axes

    @classmethod
    def update_array_shape(cls, array: np.ndarray, axes: str):
        """
        Rearrange order of array axes to get proper internal axes order

        :param array: array to reorder
        :param axes: current order of array axes as string like "TZYXC"
        """
        array, axes = cls._reduce_obsolete_dummy_axes(array, axes)

        try:
            final_mapping_dict = {letter: i for i, letter in enumerate(cls.return_order())}
            for let1, let2 in [("Z", "I"), ("Z", "Q"), ("C", "S")]:
                if let1 in final_mapping_dict and let2 not in final_mapping_dict:
                    final_mapping_dict[let2] = final_mapping_dict[let1]
            axes_li = list(axes)
            # Fixme; workaround for old saved segmentation
            if axes_li[0] == "Q" and axes_li[1] == "Q":
                axes_li[0] = "T"
                axes_li[1] = "Z"
            i = 0
            while i < len(axes_li):
                if array.shape[i] == 1:
                    array = array.take(0, i)
                    axes_li.pop(i)
                else:
                    i += 1

            final_mapping = [final_mapping_dict[letter] for letter in axes_li]
        except KeyError as e:  # pragma: no cover
            raise NotImplementedError(
                f"Data type not supported ({e.args[0]}). Please contact with author for update code"
            ) from e
        if len(final_mapping) != len(set(final_mapping)):  # pragma: no cover
            raise NotImplementedError("Data type not supported. Please contact with author for update code")
        if len(array.shape) < len(cls.return_order()):
            array = np.reshape(array, array.shape + (1,) * (len(cls.return_order()) - len(array.shape)))

        return np.moveaxis(array, list(range(len(axes_li))), final_mapping)


class BaseImageReaderBuffer(BaseImageReader):
    @abstractmethod
    def read(self, image_path: typing.Union[str, Path, BytesIO], mask_path=None, ext=None) -> Image:
        """
        Main function to read image. If ext is not set then it may be deduced from path to file.
        If BytesIO is given and non default data file type is needed then ext need to be set

        :param image_path: path to image or buffer
        :param mask_path: path to mask or buffer
        :param ext: extension if need to decide algorithm, if absent and image_path is path then
            should be deduced from path
        :return: image structure
        """
        raise NotImplementedError

    @classmethod
    def read_image(
        cls,
        image_path: typing.Union[str, Path, BytesIO],
        mask_path=None,
        callback_function: typing.Optional[typing.Callable] = None,
        default_spacing: typing.Optional[typing.Tuple[float, float, float]] = None,
    ) -> Image:
        """
        read image file with optional mask file

        :param image_path: path or opened file contains image
        :param mask_path:
        :param callback_function: function for provide information about progress in reading file (for progressbar)
        :param default_spacing: used if file do not contains information about spacing
            (or metadata format is not supported)
        :return: image
        """
        # TODO add generic description of callback function
        instance = cls(callback_function)
        if default_spacing is not None:
            instance.set_default_spacing(default_spacing)
        return instance.read(image_path, mask_path)


class GenericImageReader(BaseImageReaderBuffer):
    """This class try to decide which method use base on path"""

    def read(self, image_path: typing.Union[str, BytesIO, Path], mask_path=None, ext=None) -> Image:
        if ext is None:
            if isinstance(image_path, (str, Path)):
                ext = os.path.splitext(image_path)[1]
            else:
                ext = ".tif"
        ext = ext.lower()
        if ext == ".czi":
            return CziImageReader.read_image(image_path, mask_path, self.callback_function, self.default_spacing)
        if ext in [".oif", ".oib"]:
            if isinstance(image_path, BytesIO):
                raise NotImplementedError("Oif format is not supported for BytesIO")
            return OifImagReader.read_image(image_path, mask_path, self.callback_function, self.default_spacing)
        if ext == ".obsep":
            if isinstance(image_path, BytesIO):
                raise NotImplementedError("Obsep format is not supported for BytesIO")
            return ObsepImageReader.read_image(image_path, mask_path, self.callback_function, self.default_spacing)
        return TiffImageReader.read_image(image_path, mask_path, self.callback_function, self.default_spacing)


class OifImagReader(BaseImageReader):
    def read(self, image_path: typing.Union[str, Path], mask_path=None, ext=None) -> Image:
        with OifFile(image_path) as image_file:
            tiffs = tifffile.natural_sorted(image_file.glob("*.tif"))

            with image_file.open_file(tiffs[0]) as tiff_buffer, tifffile.TiffFile(
                tiff_buffer, name=tiffs[0]
            ) as tif_file:
                axes = image_file.series[0].axes + tif_file.series[0].axes
            image_data = image_file.asarray()
            image_data = self.update_array_shape(image_data, axes)
            with suppress(KeyError):
                self._read_scale_parameter(image_file)
                # TODO add mask reading
        return self.image_class(
            image_data,
            self.spacing,
            file_path=os.path.abspath(image_path),
            axes_order=self.return_order(),
            metadata=image_file.mainfile,
        )

    def _read_scale_parameter(self, image_file):
        flat_parm = image_file.mainfile["Reference Image Parameter"]
        x_scale = flat_parm["HeightConvertValue"] * name_to_scalar[flat_parm["HeightUnit"]]
        y_scale = flat_parm["WidthConvertValue"] * name_to_scalar[flat_parm["WidthUnit"]]
        i = 0
        while True:
            name = f"Axis {i} Parameters Common"
            if name not in image_file.mainfile:  # pragma: no cover
                z_scale = 1
                break
            axis_info = image_file.mainfile[name]
            if axis_info["AxisCode"] == "Z":
                z_scale = axis_info["Interval"] * name_to_scalar[axis_info["UnitName"]]
                break
            i += 1

        self.spacing = z_scale, x_scale, y_scale


class CziImageReader(BaseImageReaderBuffer):
    """
    This class is to read data from czi files. Masks will be treated as TIFF.
    """

    def read(self, image_path: typing.Union[str, BytesIO, Path], mask_path=None, ext=None) -> Image:
        image_file = CziFile(image_path)
        image_data = image_file.asarray(max_workers=CZI_MAX_WORKERS)
        image_data = self.update_array_shape(image_data, image_file.axes)
        metadata = image_file.metadata(False)
        with suppress(KeyError):
            scaling = metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
            scale_info = {el["Id"]: el["Value"] for el in scaling}
            self.spacing = (
                scale_info.get("Z", self.default_spacing[0]),
                scale_info.get("Y", self.default_spacing[1]),
                scale_info.get("X", self.default_spacing[2]),
            )
            channel_meta = metadata["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]
            if isinstance(channel_meta, dict):
                # single channel saved in czifile
                channel_meta = [channel_meta]
            self.channel_names = [x["Name"] for x in channel_meta]
        # TODO add mask reading
        if isinstance(image_path, BytesIO):
            image_path = ""
        image_file.close()
        return self.image_class(
            image_data,
            self.spacing,
            file_path=image_path,
            axes_order=self.return_order(),
            metadata=metadata,
            channel_names=self.channel_names,
        )

    @classmethod
    def update_array_shape(cls, array: np.ndarray, axes: str):
        if "B" in axes:
            index = axes.index("B")
            if array.shape[index] != 1:
                raise NotImplementedError(
                    "Czi file with B axes is not currently supported by PartSeg."
                    " Please contact with author for update code"
                )
            array = array.take(0, axis=index)
            axes = axes[:index] + axes[index + 1 :]
        if axes[-1] == "0":
            array = array[..., 0]
            axes = axes[:-1]
        return super().update_array_shape(array, axes)


class ObsepImageReader(BaseImageReader):
    def _search_for_files(
        self,
        directory: Path,
        channels: typing.List["Element"],
        suffix: str = "",
        required: bool = False,
    ) -> typing.List[Image]:
        possible_extensions = [".tiff", ".tif", ".TIFF", ".TIF"]
        channel_list = []
        for channel in channels:
            try:
                name = next(iter(channel)).attrib["val"] + suffix
            except StopIteration as e:  # pragma: no cover
                raise ValueError("Missed information about channel name in obsep file") from e
            for ex in possible_extensions:
                if (directory / (name + ex)).exists():
                    name += ex
                    break
            else:  # pragma: no cover
                if required:
                    raise ValueError(f"Not found file for key {name}")
                continue
            channel_list.append(TiffImageReader.read_image(directory / name, default_spacing=self.default_spacing))
        return channel_list

    def read(self, image_path: typing.Union[str, Path], mask_path=None, ext=None) -> Image:
        directory = Path(os.path.dirname(image_path))
        xml_doc = ElementTree.parse(image_path).getroot()
        channels = xml_doc.findall("net/node/node/attribute[@name='image type']")
        if not channels:
            raise ValueError("Information about channel images not found")  # pragma: no cover
        channel_list = [
            *self._search_for_files(directory, channels, required=True),
            *self._search_for_files(directory, channels, "_deconv"),
        ]
        image = channel_list[0]
        for el in channel_list[1:]:
            image = image.merge(el, "C")

        z_spacing = (
            float(xml_doc.find("net/node/attribute[@name='step width']/double").attrib["val"]) * name_to_scalar["um"]
        )

        image.set_spacing((z_spacing,) + image.spacing[1:])
        image.file_path = str(image_path)
        return image


class TiffImageReader(BaseImageReaderBuffer):
    """
    TIFF/LSM files reader. Base reading with :py:meth:`BaseImageReader.read_image`

    image_file: tifffile.TiffFile
    mask_file: tifffile.TiffFile
    """

    def __init__(self, callback_function=None):
        super().__init__(callback_function)
        self.colors = None
        self.ranges = None
        self.shift = (0, 0, 0)
        self.name = ""
        self.metadata = {}

    def read(self, image_path: typing.Union[str, BytesIO, Path], mask_path=None, ext=None) -> Image:
        """
        Read tiff image from tiff_file
        """
        self.spacing, self.colors, self.channel_names, self.ranges = self.default_spacing, None, None, None
        with tifffile.TiffFile(image_path) as image_file:
            total_pages_num = len(image_file.series[0])

            axes = image_file.series[0].axes

            if image_file.is_lsm:
                self.read_lsm_metadata(image_file)
            elif image_file.is_imagej:
                self.read_imagej_metadata(image_file)
            elif image_file.is_ome:
                self.read_ome_metadata(image_file)
            else:
                x_spacing, y_spacing = self.read_resolution_from_tags(image_file)
                self.spacing = self.default_spacing[0], y_spacing, x_spacing
            mutex = Lock()
            count_pages = [0]

            def report_func():
                mutex.acquire()
                count_pages[0] += 1
                self.callback_function("step", count_pages[0])
                mutex.release()

            if mask_path is not None:
                with tifffile.TiffFile(mask_path) as mask_file:
                    self.callback_function("max", total_pages_num + len(mask_file.series[0]))
                    self.verify_mask(mask_file, image_file)
                    mask_file.report_func = report_func
                    mask_data = mask_file.asarray()
                    mask_data = self.update_array_shape(mask_data, mask_file.series[0].axes)
                    if "C" in self.return_order():
                        pos: typing.List[typing.Union[slice, int]] = [slice(None) for _ in range(mask_data.ndim)]
                        pos[self.return_order().index("C")] = 0
                        mask_data = mask_data[tuple(pos)]

            else:
                mask_data = None
                if total_pages_num > 1:
                    self.callback_function("max", total_pages_num)

            image_file.report_func = report_func
            try:
                image_data = image_file.asarray()
            except ValueError as e:  # pragma: no cover
                raise TiffFileException(*e.args) from e
            image_data = self.update_array_shape(image_data, axes)

        if not isinstance(image_path, (str, Path)):
            image_path = ""
        return self.image_class(
            image_data,
            self.spacing,
            mask=mask_data,
            default_coloring=self.colors,
            channel_names=self.channel_names,
            ranges=self.ranges,
            file_path=os.path.abspath(image_path),
            axes_order=self.return_order(),
            shift=self.shift,
            name=self.name,
            metadata=self.metadata,
        )

    @staticmethod
    def verify_mask(mask_file, image_file):
        """
        verify if mask fit to image. Raise ValueError exception on error
        :return:
        """
        if mask_file is None:  # pragma: no cover
            return
        image_series = image_file.pages[0]
        mask_series = mask_file.pages[0]
        for i, pos in enumerate(mask_series.axes):
            if mask_series.shape[i] == 1:  # pragma: no cover
                continue
            try:
                j = image_series.axes.index(pos)
            except ValueError as e:  # pragma: no cover
                raise ValueError(f"{INCOMPATIBLE_IMAGE_MASK} (axes)") from e
                # TODO add verification if problem with T/Z/I
            if image_series.shape[j] != mask_series.shape[i]:  # pragma: no cover
                raise ValueError(INCOMPATIBLE_IMAGE_MASK)
            # TODO Add verification if mask have to few dimensions

    @staticmethod
    def decode_int(val: int):
        """
        This function split 32 bits int on 4 8-bits ints

        :param val: value to decode
        :return: list of four numbers with values from [0, 255]
        """
        return [(val >> x) & 255 for x in [24, 16, 8, 0]]

    def read_resolution_from_tags(self, image_file):
        tags = image_file.pages[0].tags
        try:
            if image_file.is_imagej:
                scalar = name_to_scalar[image_file.imagej_metadata["unit"]]
            else:
                unit = tags["ResolutionUnit"].value
                if unit == 3:
                    scalar = name_to_scalar["centimeter"]
                elif unit == 2:
                    scalar = name_to_scalar["cal"]
                else:  # pragma: no cover
                    raise KeyError(f"wrong scalar {tags['ResolutionUnit']}, {tags['ResolutionUnit'].value}")

            x_spacing = tags["XResolution"].value[1] / tags["XResolution"].value[0] * scalar
            y_spacing = tags["YResolution"].value[1] / tags["YResolution"].value[0] * scalar
        except (KeyError, ZeroDivisionError):
            x_spacing, y_spacing = self.default_spacing[2], self.default_spacing[1]
        return x_spacing, y_spacing

    def read_imagej_metadata(self, image_file):
        try:
            z_spacing = image_file.imagej_metadata["spacing"] * name_to_scalar[image_file.imagej_metadata["unit"]]
        except KeyError:
            z_spacing = self.default_spacing[0]
        x_spacing, y_spacing = self.read_resolution_from_tags(image_file)
        self.spacing = z_spacing, y_spacing, x_spacing
        self.colors = image_file.imagej_metadata.get("LUTs")
        self.channel_names = image_file.imagej_metadata.get("Labels")
        if "Ranges" in image_file.imagej_metadata:
            ranges = image_file.imagej_metadata["Ranges"]
            self.ranges = list(zip(ranges[::2], ranges[1::2]))
        self.metadata = image_file.imagej_metadata

    def _read_ome_channel_information(self, meta_data):
        if "Channel" not in meta_data["Pixels"]:
            return
        if isinstance(meta_data["Pixels"]["Channel"], (list, tuple)):
            with suppress(KeyError):
                self.channel_names = [ch["Name"] for ch in meta_data["Pixels"]["Channel"]]
            with suppress(KeyError):
                self.colors = [self.decode_int(ch["Color"])[:-1] for ch in meta_data["Pixels"]["Channel"]]
            return
        if "Name" in meta_data["Pixels"]["Channel"]:
            self.channel_names = [meta_data["Pixels"]["Channel"]["Name"]]
        if "Color" in meta_data["Pixels"]["Channel"]:
            self.channel_names = [meta_data["Pixels"]["Channel"]["Color"]]

    def read_ome_metadata(self, image_file):
        meta_data = tifffile.xml2dict(image_file.ome_metadata)["OME"]["Image"]
        with suppress(KeyError):
            self.spacing = [
                meta_data["Pixels"][f"PhysicalSize{x}"] * name_to_scalar[meta_data["Pixels"][f"PhysicalSize{x}Unit"]]
                for x in ["Z", "Y", "X"]
            ]
        with suppress(KeyError):
            self.shift = [
                meta_data["Pixels"]["Plane"][0][f"Position{x}"]
                * name_to_scalar[meta_data["Pixels"]["Plane"][0][f"Position{x}Unit"]]
                for x in ["Z", "Y", "X"]
            ]
        self.metadata = meta_data
        self.name = meta_data.get("Name", "")
        self._read_ome_channel_information(meta_data)

    def read_lsm_metadata(self, image_file):
        self.spacing = [image_file.lsm_metadata[f"VoxelSize{x}"] for x in ["Z", "Y", "X"]]
        if "ChannelColors" in image_file.lsm_metadata:
            if "Colors" in image_file.lsm_metadata["ChannelColors"]:
                self.colors = [x[:3] for x in image_file.lsm_metadata["ChannelColors"]["Colors"]]
            if "ColorNames" in image_file.lsm_metadata["ChannelColors"]:
                self.channel_names = image_file.lsm_metadata["ChannelColors"]["ColorNames"]
        self.metadata = image_file.lsm_metadata


name_to_scalar = {
    "micron": 10**-6,
    "µm": 10**-6,
    "um": 10**-6,
    "nm": 10**-9,
    "mm": 10**-3,
    "millimeter": 10**-3,
    "pm": 10**-12,
    "picometer": 100**-12,
    "nanometer": 10**-9,
    "\\u00B5m": 10**-6,
    "centimeter": 10**-2,
    "cm": 10**-2,
    "cal": 2.54 * 10**-2,
}  #: dict with known names of scalar to scalar value. Some may be  missed
