import os.path
import typing
import warnings
from abc import abstractmethod
from io import BytesIO
from pathlib import Path
from threading import Lock

import numpy as np
import tifffile.tifffile
from czifile.czifile import CziFile
from defusedxml import ElementTree
from oiffile import OifFile
from tifffile import TiffFile

from .image import Image


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
        if hasattr(cls.image_class, "return_order"):  # pragma: no cover
            warnings.warn(
                "Using return_order is deprecated since PartSeg 0.12.0. Please fix your image_class", DeprecationWarning
            )
            return cls.image_class.return_order
        return cls.image_class.axis_order

    def __init__(self, callback_function=None):
        self.default_spacing = 10 ** -6, 10 ** -6, 10 ** -6
        self.spacing = self.default_spacing
        self.image_file = None
        if callback_function is None:
            self.callback_function = lambda x, y: 0
        else:
            self.callback_function = callback_function

    def set_default_spacing(self, spacing):
        spacing = tuple(spacing)
        if len(spacing) == 2:
            # one micrometer
            spacing = (10 ** -6,) + spacing
        if len(spacing) != 3:
            raise ValueError(f"wrong spacing {spacing}")
        self.default_spacing = spacing

    @abstractmethod
    def read(self, image_path: typing.Union[str, BytesIO, Path], mask_path=None, ext=None) -> Image:
        """
        Main function to read image. If ext is not set then it may be deduced from path to file.
        If BytesIO is given and non default data file type is needed then ext need to be set

        :param image_path: path to image or buffer
        :param mask_path: path to mask or buffer
        :param ext: extension if need to decide algorithm, if absent and image_path is path then
            should be deduced from path
        :return: image structure
        """
        raise NotImplementedError()

    @classmethod
    def read_image(
        cls,
        image_path: typing.Union[str, BytesIO, Path],
        mask_path=None,
        callback_function: typing.Optional[typing.Callable] = None,
        default_spacing: typing.Tuple[float, float, float] = None,
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

    @classmethod
    def update_array_shape(cls, array: np.ndarray, axes: str):
        """
        Rearrange order of array axes to get proper internal axes order

        :param array: array to reorder
        :param axes: current order of array axes as string like "TZYXC"
        """
        try:
            final_mapping_dict = {l: i for i, l in enumerate(cls.return_order())}
            if "Z" in final_mapping_dict and "I" not in final_mapping_dict:
                final_mapping_dict["I"] = final_mapping_dict["Z"]
            if "Z" in final_mapping_dict and "Q" not in final_mapping_dict:
                final_mapping_dict["Q"] = final_mapping_dict["Z"]
            if "C" in final_mapping_dict and "S" not in final_mapping_dict:
                final_mapping_dict["S"] = final_mapping_dict["C"]
            axes = list(axes)
            # Fixme; workaround for old saved segmentation
            if axes[0] == "Q" and axes[1] == "Q":
                axes[0] = "T"
                axes[1] = "Z"
            i = 0
            while i < len(axes):
                name = axes[i]
                if name not in final_mapping_dict and array.shape[i] == 1:
                    array = array.take(0, i)
                    axes.pop(i)
                else:
                    i += 1

            final_mapping = [final_mapping_dict[letter] for letter in axes]
        except KeyError as e:  # pragma: no cover
            raise NotImplementedError(
                f"Data type not supported ({e.args[0]}). Please contact with author for update code"
            )
        if len(final_mapping) != len(set(final_mapping)):
            raise NotImplementedError("Data type not supported. Please contact with author for update code")
        if len(array.shape) < len(cls.return_order()):
            array = np.reshape(array, array.shape + (1,) * (len(cls.return_order()) - len(array.shape)))

        array = np.moveaxis(array, list(range(len(axes))), final_mapping)
        return array


class GenericImageReader(BaseImageReader):
    """This class try to decide which method use base on path """

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
            return OifImagReader.read_image(image_path, mask_path, self.callback_function, self.default_spacing)
        if ext == ".obsep":
            return ObsepImageReader.read_image(image_path, mask_path, self.callback_function, self.default_spacing)
        return TiffImageReader.read_image(image_path, mask_path, self.callback_function, self.default_spacing)


class OifImagReader(BaseImageReader):
    def read(self, image_path: typing.Union[str, BytesIO, Path], mask_path=None, ext=None) -> Image:
        self.image_file = OifFile(image_path)
        tiffs = self.image_file.tiffs
        tif_file = TiffFile(self.image_file.open_file(tiffs.files[0]), name=tiffs.files[0])
        axes = tiffs.axes + tif_file.series[0].axes
        image_data = self.image_file.asarray()
        image_data = self.update_array_shape(image_data, axes)
        try:
            flat_parm = self.image_file.mainfile["Reference Image Parameter"]
            x_scale = flat_parm["HeightConvertValue"] * name_to_scalar[flat_parm["HeightUnit"]]
            y_scale = flat_parm["WidthConvertValue"] * name_to_scalar[flat_parm["WidthUnit"]]
            i = 0
            while True:
                name = f"Axis {i} Parameters Common"
                if name not in self.image_file.mainfile:
                    z_scale = 1
                    break
                axis_info = self.image_file.mainfile[name]
                if axis_info["AxisCode"] == "Z":
                    z_scale = axis_info["Interval"] * name_to_scalar[axis_info["UnitName"]]
                    break
                i += 1

            self.spacing = z_scale, x_scale, y_scale
        except KeyError:  # pragma: no cover
            pass
        # TODO add mask reading
        return self.image_class(
            image_data, self.spacing, file_path=os.path.abspath(image_path), axes_order=self.return_order()
        )


class CziImageReader(BaseImageReader):
    """
    This class is to read data from czi files. Masks will be treated as TIFF.
    """

    def read(self, image_path: typing.Union[str, BytesIO, Path], mask_path=None, ext=None) -> Image:
        self.image_file = CziFile(image_path)
        image_data = self.image_file.asarray()
        image_data = self.update_array_shape(image_data, self.image_file.axes)
        metadata = self.image_file.metadata(False)
        try:
            scaling = metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
            scale_info = {el["Id"]: el["Value"] for el in scaling}
            self.spacing = (
                scale_info.get("Z", self.default_spacing[0]),
                scale_info.get("Y", self.default_spacing[1]),
                scale_info.get("X", self.default_spacing[2]),
            )
        except KeyError:  # pragma: no cover
            pass
        # TODO add mask reading
        return self.image_class(
            image_data, self.spacing, file_path=os.path.abspath(image_path), axes_order=self.return_order()
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
    def read(self, image_path: typing.Union[str, BytesIO, Path], mask_path=None, ext=None) -> Image:
        directory = Path(os.path.dirname(image_path))
        xml_doc = ElementTree.parse(image_path).getroot()
        channels = xml_doc.findall("net/node/node/attribute[@name='image type']")
        if not channels:
            raise ValueError("Information about channel images not found")
        possible_extensions = [".tiff", ".tif", ".TIFF", ".TIF"]
        channel_list = []
        for channel in channels:
            try:
                name = next(iter(channel)).attrib["val"]
            except StopIteration:  # pragma: no cover
                raise ValueError("Missed information about channel name in obsep file")
            for ex in possible_extensions:
                if (directory / (name + ex)).exists():
                    name += ex
                    break
            else:  # pragma: no cover
                raise ValueError(f"Not found file for key {name}")
            channel_list.append(TiffImageReader.read_image(directory / name, default_spacing=self.default_spacing))
        for channel in channels:
            try:
                name = next(iter(channel)).attrib["val"] + "_deconv"
            except StopIteration:  # pragma: no cover
                raise ValueError("Missed information about channel name in obsep file")
            for ex in possible_extensions:
                if (directory / (name + ex)).exists():
                    name += ex
                    break
            if (directory / name).exists():
                channel_list.append(TiffImageReader.read_image(directory / name, default_spacing=self.default_spacing))

        image = channel_list[0]
        for el in channel_list[1:]:
            image = image.merge(el, "C")

        z_spacing = (
            float(xml_doc.find("net/node/attribute[@name='step width']/double").attrib["val"]) * name_to_scalar["um"]
        )

        image.set_spacing((z_spacing,) + image.spacing[1:])
        image.file_path = str(image_path)
        return image


class TiffImageReader(BaseImageReader):
    """
    TIFF/LSM files reader. Base reading with :py:meth:`BaseImageReader.read_image`

    image_file: TiffFile
    mask_file: TiffFile
    """

    def __init__(self, callback_function=None):
        super().__init__(callback_function)
        self.image_file = None
        self.mask_file: typing.Optional[TiffFile] = None
        self.colors = None
        self.channel_names = None
        self.ranges = None

    def read(self, image_path: typing.Union[str, BytesIO, Path], mask_path=None, ext=None) -> Image:
        """
        Read tiff image from tiff_file
        """
        self.spacing, self.colors, self.channel_names, self.ranges = self.default_spacing, None, None, None
        self.image_file = TiffFile(image_path)
        total_pages_num = len(self.image_file.series[0])
        if mask_path is not None:
            self.mask_file = TiffFile(mask_path)
            total_pages_num += len(self.mask_file.series[0])
            self.verify_mask()
        else:
            self.mask_file = None

        # shape = self.image_file.series[0].shape
        axes = self.image_file.series[0].axes
        self.callback_function("max", total_pages_num)

        if self.image_file.is_lsm:
            self.read_lsm_metadata()
        elif self.image_file.is_imagej:
            self.read_imagej_metadata()
        elif self.image_file.is_ome:
            self.read_ome_metadata()
        else:
            x_spac, y_spac = self.read_resolution_from_tags()
            self.spacing = self.default_spacing[0], y_spac, x_spac
        mutex = Lock()
        count_pages = [0]

        def report_func():
            mutex.acquire()
            count_pages[0] += 1
            self.callback_function("step", count_pages[0])
            mutex.release()

        self.image_file.report_func = report_func
        try:
            image_data = self.image_file.asarray()
        except ValueError as e:  # pragma: no cover
            raise TiffFileException(*e.args)
        image_data = self.update_array_shape(image_data, axes)
        if self.mask_file is not None:
            self.mask_file.report_func = report_func
            mask_data = self.mask_file.asarray()
            mask_data = self.update_array_shape(mask_data, self.mask_file.series[0].axes)[..., 0]
        else:
            mask_data = None
        self.image_file.close()
        if self.mask_file is not None:
            self.mask_file.close()
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
        )

    def verify_mask(self):
        """
        verify if mask fit to image. Raise ValueError exception on error
        :return:
        """
        if self.mask_file is None:  # pragma: no cover
            return
        image_series = self.image_file.pages[0]
        mask_series = self.mask_file.pages[0]
        for i, pos in enumerate(mask_series.axes):
            if mask_series.shape[i] == 1:  # pragma: no cover
                continue
            try:
                j = image_series.axes.index(pos)
            except ValueError:  # pragma: no cover
                raise ValueError("Incompatible shape of mask and image (axes)")
                # TODO add verification if problem with T/Z/I
            if image_series.shape[j] != mask_series.shape[i]:  # pragma: no cover
                raise ValueError("Incompatible shape of mask and image")
            # TODO Add verification if mask have to few dimensions

    @staticmethod
    def decode_int(val: int):
        """
        This function split 32 bits int on 4 8-bits ints

        :param val: value to decode
        :return: list of four numbers with values from [0, 255]
        """
        return [(val >> x) & 255 for x in [24, 16, 8, 0]]

    def read_resolution_from_tags(self):
        tags = self.image_file.pages[0].tags
        try:

            if self.image_file.is_imagej:
                scalar = name_to_scalar[self.image_file.imagej_metadata["unit"]]
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

    def read_imagej_metadata(self):
        try:
            z_spacing = (
                self.image_file.imagej_metadata["spacing"] * name_to_scalar[self.image_file.imagej_metadata["unit"]]
            )
        except KeyError:
            z_spacing = self.default_spacing[0]
        x_spacing, y_spacing = self.read_resolution_from_tags()
        self.spacing = z_spacing, y_spacing, x_spacing
        self.colors = self.image_file.imagej_metadata.get("LUTs")
        self.channel_names = self.image_file.imagej_metadata.get("Labels")
        if "Ranges" in self.image_file.imagej_metadata:
            ranges = self.image_file.imagej_metadata["Ranges"]
            self.ranges = list(zip(ranges[::2], ranges[1::2]))

    def read_ome_metadata(self):
        if isinstance(self.image_file.ome_metadata, str):
            if hasattr(tifffile, "xml2dict"):
                meta_data = tifffile.xml2dict(self.image_file.ome_metadata)["OME"]["Image"]["Pixels"]
            else:
                return
        else:
            meta_data = self.image_file.ome_metadata["Image"]["Pixels"]
        try:
            self.spacing = [
                meta_data[f"PhysicalSize{x}"] * name_to_scalar[meta_data[f"PhysicalSize{x}Unit"]]
                for x in ["Z", "Y", "X"]
            ]
        except KeyError:  # pragma: no cover
            pass
        if "Channel" in meta_data and isinstance(meta_data["Channel"], (list, tuple)):
            try:
                self.channel_names = [ch["Name"] for ch in meta_data["Channel"]]
            except KeyError:
                pass
            try:
                self.colors = [self.decode_int(ch["Color"])[:-1] for ch in meta_data["Channel"]]
            except KeyError:
                pass

    def read_lsm_metadata(self):
        self.spacing = [self.image_file.lsm_metadata[f"VoxelSize{x}"] for x in ["Z", "Y", "X"]]
        if "ChannelColors" in self.image_file.lsm_metadata:
            if "Colors" in self.image_file.lsm_metadata["ChannelColors"]:
                self.colors = [x[:3] for x in self.image_file.lsm_metadata["ChannelColors"]["Colors"]]
            if "ColorNames" in self.image_file.lsm_metadata["ChannelColors"]:
                self.channel_names = self.image_file.lsm_metadata["ChannelColors"]["ColorNames"]


name_to_scalar = {
    "micron": 10 ** -6,
    "µm": 10 ** -6,
    "um": 10 ** -6,
    "nm": 10 ** -9,
    "mm": 10 ** -3,
    "millimeter": 10 ** -3,
    "pm": 10 ** -12,
    "picometer": 100 ** -12,
    "nanometer": 10 ** -9,
    "\\u00B5m": 10 ** -6,
    "centimeter": 10 ** -2,
    "cm": 10 ** -2,
    "cal": 2.54 * 10 ** -2,
}  #: dict with known names of scalar to scalar value. May be some missed
