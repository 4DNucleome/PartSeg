import typing
from io import BytesIO
from threading import Lock

from tifffile import TiffFile
from tifffile.tifffile import TiffPage
import tifffile.tifffile

from .image import Image
import numpy as np
import os.path


class ImageReader(object):
    """
    image_file: TiffFile
    mask_file: TiffFile
    """

    def __init__(self, callback_function=None):
        self.image_file = None
        self.mask_file: TiffFile = None
        self.default_spacing = 10**-6, 10**-6, 10**-6
        self.spacing = self.default_spacing
        self.colors = None
        self.labels = None
        self.ranges = None
        if callback_function is None:
            self.callback_function = lambda x, y: 0
        else:
            self.callback_function = callback_function

    def set_default_spacing(self, spacing):
        spacing = tuple(spacing)
        if len(spacing) == 2:
            spacing = (1,) + spacing
        if len(spacing) != 3:
            raise ValueError(f"wrong spacing {spacing}")
        self.default_spacing = spacing


    @classmethod
    def read_image(cls, image_path: typing.Union[str, BytesIO], mask_path=None,
                   callback_function: typing.Optional[typing.Callable]=None,
                   default_spacing: typing.List[int]=None) -> Image:
        instance = cls(callback_function)
        if default_spacing is not None:
            instance.set_default_spacing(default_spacing)
        return instance.read(image_path, mask_path)


    def read(self, image_path: typing.Union[str, BytesIO], mask_path=None) -> Image:
        """
        Read tiff image from tiff_file

        :param image_path:
        :param mask_path:
        :return: Image
        """
        self.spacing, self.colors, self.labels, self.ranges, order = self.default_spacing, None, None, None, None
        self.image_file = TiffFile(image_path)
        total_pages_num = len(self.image_file.series[0])
        if mask_path is not None:
            self.mask_file = TiffFile(mask_path)
            total_pages_num += len(self.mask_file.series[0])
            self.verify_mask()
        else:
            self.mask_file = None

        shape = self.image_file.series[0].shape
        axes = self.image_file.series[0].axes
        self.callback_function("max", total_pages_num)

        if self.image_file.is_lsm:
            self.read_lsm_metadata()
        elif self.image_file.is_imagej:
            self.read_imagej_metadata()
        elif self.image_file.is_ome:
            self.read_ome_metadata()
        mutex = Lock()
        count_pages = [0]

        def report_func():
            mutex.acquire()
            count_pages[0] += 1
            self.callback_function("step", count_pages[0])
            mutex.release()

        self.image_file.report_func = report_func
        image_data = self.image_file.asarray()
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
        if not isinstance(image_path, str):
            image_path = ""
        return Image(image_data, self.spacing, mask=mask_data, default_coloring=self.colors, labels=self.labels,
                     ranges=self.ranges, file_path=os.path.abspath(image_path))

    @staticmethod
    def update_array_shape(array: np.ndarray, axes: str):
        try:
            final_mapping_dict = {"T": 0, "Z": 1, "Y": 2, "X": 3, "C": 4, "I": 1, "S": 4, "Q": 1}
            final_mapping = [final_mapping_dict[letter] for letter in axes]
        except KeyError:
            raise NotImplementedError("Data type not supported. Please contact with author for update code")
        if len(final_mapping) != len(set(final_mapping)):
            raise NotImplementedError("Data type not supported. Please contact with author for update code")
        if len(array.shape) < 5:
            array = np.reshape(array, array.shape + (1,) * (5 - len(array.shape)))

        array = np.moveaxis(array, list(range(len(axes))), final_mapping)
        return array

    def verify_mask(self):
        if self.mask_file is None:
            return
        image_series = self.image_file.pages[0]
        mask_series = self.mask_file.pages[0]
        for i, pos in enumerate(mask_series.axes):
            if mask_series.shape[i] == 1:
                continue
            try:
                j = image_series.axes.index(pos)
            except ValueError:
                raise ValueError("Incompatible shape of mask and image (axes)")
                # TODO add verification if problem with T/Z/I
            if image_series.shape[j] != mask_series.shape[i]:
                raise ValueError("Incompatible shape of mask and image")
            # TODO Add verification if mask have to few dimensions

    @staticmethod
    def decode_int(val: int):
        return [(val >> x) & 255 for x in [24, 16, 8, 0]]

    def read_imagej_metadata(self):
        assert self.image_file.is_imagej
        try:
            z_spacing = \
                self.image_file.imagej_metadata["spacing"] * name_to_scalar[self.image_file.imagej_metadata["unit"]]
        except KeyError:
            z_spacing = 1
        tags = self.image_file.pages[0].tags
        try:
            x_spacing = tags["XResolution"].value[1] / tags["XResolution"].value[0] \
                * name_to_scalar[self.image_file.imagej_metadata["unit"]]
            y_spacing = tags["YResolution"].value[1] / tags["YResolution"].value[0] \
                * name_to_scalar[self.image_file.imagej_metadata["unit"]]
        except KeyError:
            x_spacing, y_spacing = 1, 1
        self.spacing = z_spacing, x_spacing, y_spacing
        self.colors = self.image_file.imagej_metadata.get("LUTs")
        self.labels = self.image_file.imagej_metadata.get("Labels")
        if "Ranges" in self.image_file.imagej_metadata:
            ranges = self.image_file.imagej_metadata["Ranges"]
            self.ranges = list(zip(ranges[::2], ranges[1::2]))

    def read_ome_metadata(self):
        meta_data = self.image_file.ome_metadata['Image']["Pixels"]
        try:
            self.spacing = [meta_data[f"PhysicalSize{x}"] *
                            name_to_scalar[meta_data[f"PhysicalSize{x}Unit"]] for x in ["Z", "Y", "X"]]
        except KeyError:
            pass
        if "Channel" in meta_data and isinstance(meta_data["Channel"], (list, tuple)):
            try:
                self.labels = [ch["Name"] for ch in meta_data["Channel"]]
            except KeyError:
                pass
            try:
                self.colors = [self.decode_int(ch["Color"])[:-1] for ch in meta_data["Channel"]]
            except KeyError:
                pass

    def read_lsm_metadata(self):
        self.spacing = \
            [self.image_file.lsm_metadata[f"VoxelSize{x}"] for x in ["Z", "Y", "X"]]
        if "ChannelColors" in self.image_file.lsm_metadata:
            if "Colors" in self.image_file.lsm_metadata["ChannelColors"]:
                self.colors = [x[:3] for x in self.image_file.lsm_metadata["ChannelColors"]["Colors"]]
            if "ColorNames" in self.image_file.lsm_metadata["ChannelColors"]:
                self.labels = self.image_file.lsm_metadata["ChannelColors"]["ColorNames"]


def increment_dict(dkt: dict):
    for el in dkt:
        dkt[el] += 1


name_to_scalar = {
    "micron": 10 ** -6,
    "µm": 10 ** -6,
    "\\u00B5m": 10 ** -6
}


class MyTiffPage(TiffPage):
    def asarray(self, *args, **kwargs):
        # Because of TiffFrame usage
        res = TiffPage.asarray(self, *args, **kwargs)
        self.parent.report_func()
        return res


TiffFile.report_func = lambda x: 0

tifffile.tifffile.TiffPage = MyTiffPage

if tifffile.__version__ == '0.15.1':
    import warnings
    import numpy
    from tifffile import product, squeeze_axes
    from tifffile.tifffile import TIFF, TiffPageSeries
    def _ome_series(self):
        """Return image series in OME-TIFF file(s)."""
        from xml.etree import cElementTree as etree  # delayed import
        omexml = self.pages[0].description
        try:
            root = etree.fromstring(omexml)
        except etree.ParseError as e:
            # TODO: test badly encoded OME-XML
            warnings.warn('ome-xml: %s' % e)
            try:
                # might work on Python 2
                omexml = omexml.decode('utf-8', 'ignore').encode('utf-8')
                root = etree.fromstring(omexml)
            except Exception:
                return

        self.pages.useframes = True
        self.pages.keyframe = 0
        self.pages.load()

        uuid = root.attrib.get('UUID', None)
        self._files = {uuid: self}
        dirname = self._fh.dirname
        modulo = {}
        series = []
        for element in root:
            if element.tag.endswith('BinaryOnly'):
                # TODO: load OME-XML from master or companion file
                warnings.warn('ome-xml: not an ome-tiff master file')
                break
            if element.tag.endswith('StructuredAnnotations'):
                for annot in element:
                    if not annot.attrib.get('Namespace',
                                            '').endswith('modulo'):
                        continue
                    for value in annot:
                        for modul in value:
                            for along in modul:
                                if not along.tag[:-1].endswith('Along'):
                                    continue
                                axis = along.tag[-1]
                                newaxis = along.attrib.get('Type', 'other')
                                newaxis = TIFF.AXES_LABELS[newaxis]
                                if 'Start' in along.attrib:
                                    step = float(along.attrib.get('Step', 1))
                                    start = float(along.attrib['Start'])
                                    stop = float(along.attrib['End']) + step
                                    labels = numpy.arange(start, stop, step)
                                else:
                                    labels = [label.text for label in along
                                              if label.tag.endswith('Label')]
                                modulo[axis] = (newaxis, labels)

            if not element.tag.endswith('Image'):
                continue

            attr = element.attrib
            name = attr.get('Name', None)

            for pixels in element:
                if not pixels.tag.endswith('Pixels'):
                    continue
                attr = pixels.attrib
                dtype = attr.get('PixelType', None)
                axes = ''.join(reversed(attr['DimensionOrder']))
                shape = idxshape = list(int(attr['Size' + ax]) for ax in axes)
                size = product(shape[:-2])
                ifds = None
                spp = 1  # samples per pixel
                # FIXME: this implementation assumes the last two
                # dimensions are stored in tiff pages (shape[:-2]).
                # Apparently that is not always the case.
                for data in pixels:
                    if data.tag.endswith('Channel'):
                        attr = data.attrib
                        if ifds is None:
                            spp = int(attr.get('SamplesPerPixel', spp))
                            ifds = [None] * (size // spp)
                            if spp > 1:
                                # correct channel dimension for spp
                                idxshape = list((shape[i] // spp if ax == 'C'
                                                 else shape[i])
                                                for i, ax in enumerate(axes))
                        elif int(attr.get('SamplesPerPixel', 1)) != spp:
                            raise ValueError(
                                "cannot handle differing SamplesPerPixel")
                        continue
                    if ifds is None:
                        ifds = [None] * (size // spp)
                    if not data.tag.endswith('TiffData'):
                        continue
                    attr = data.attrib
                    ifd = int(attr.get('IFD', 0))
                    num = int(attr.get('NumPlanes', 1 if 'IFD' in attr else 0))
                    num = int(attr.get('PlaneCount', num))
                    idx = [int(attr.get('First' + ax, 0)) for ax in axes[:-2]]
                    try:
                        idx = numpy.ravel_multi_index(idx, idxshape[:-2])
                    except ValueError:
                        # ImageJ produces invalid ome-xml when cropping
                        warnings.warn('ome-xml: invalid TiffData index')
                        continue
                    for uuid in data:
                        if not uuid.tag.endswith('UUID'):
                            continue
                        if uuid.text not in self._files:
                            if not self._multifile:
                                # abort reading multifile OME series
                                # and fall back to generic series
                                return []
                            fname = uuid.attrib['FileName']
                            try:
                                tif = TiffFile(os.path.join(dirname, fname))
                                tif.pages.useframes = True
                                tif.pages.keyframe = 0
                                tif.pages.load()
                            except (IOError, FileNotFoundError, ValueError):
                                warnings.warn(
                                    "ome-xml: failed to read '%s'" % fname)
                                break
                            self._files[uuid.text] = tif
                            tif.close()
                        pages = self._files[uuid.text].pages
                        try:
                            for i in range(num if num else len(pages)):
                                ifds[idx + i] = pages[ifd + i]
                        except IndexError:
                            warnings.warn('ome-xml: index out of range')
                        # only process first UUID
                        break
                    else:
                        pages = self.pages
                        try:
                            for i in range(num if num else len(pages)):
                                ifds[idx + i] = pages[ifd + i]
                        except IndexError:
                            warnings.warn('ome-xml: index out of range')

                if all(i is None for i in ifds):
                    # skip images without data
                    continue

                # set a keyframe on all IFDs
                keyframe = None
                for i in ifds:
                    # try find a TiffPage
                    if i and i == i.keyframe:
                        keyframe = i
                        break
                if not keyframe:
                    # reload a TiffPage from file
                    for i, keyframe in enumerate(ifds):
                        if keyframe:
                            keyframe.parent.pages.keyframe = keyframe.index
                            keyframe = keyframe.parent.pages[keyframe.index]
                            ifds[i] = keyframe
                            break
                for i in ifds:
                    if i is not None:
                        i.keyframe = keyframe

                dtype = keyframe.dtype
                series.append(
                    TiffPageSeries(ifds, shape, dtype, axes, parent=self,
                                   name=name, stype='OME'))
        for serie in series:
            shape = list(serie.shape)
            for axis, (newaxis, labels) in modulo.items():
                i = serie.axes.index(axis)
                size = len(labels)
                if shape[i] == size:
                    serie.axes = serie.axes.replace(axis, newaxis, 1)
                else:
                    shape[i] //= size
                    shape.insert(i + 1, size)
                    serie.axes = serie.axes.replace(axis, axis + newaxis, 1)
            serie.shape = tuple(shape)
        # squeeze dimensions
        for serie in series:
            serie.shape, serie.axes = squeeze_axes(serie.shape, serie.axes)
        return series


    tifffile.tifffile.TiffFile._ome_series = _ome_series
