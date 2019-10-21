import collections
import re
import os


import packaging.version
import tifffile.tifffile
from tifffile import TiffPage, TiffFile

if tifffile.tifffile.TiffPage.__module__ != "PartSegImage.tifffile_fixes":
    class MyTiffPage(TiffPage):
        """Modification of :py:class:`TiffPage` from `tifffile` package to provide progress information"""
        def asarray(self, *args, **kwargs):
            """
            Modified for progress info. call original implementation and send info that page is read by
            call parameter less function `report_func` of parent.
            sample of usage in :py:meth:`ImageRead.read`
            """
            # Because of TiffFrame usage
            res = TiffPage.asarray(self, *args, **kwargs)
            self.parent.report_func()
            return res

        @property
        def is_ome2(self):
            """Page contains OME-XML in ImageDescription tag."""
            if self.index > 1 or not self.description:
                return False
            d = self.description
            return (d[:14] == '<?xml version=' or d[:15] == '<?xml version =') and \
                   (d[-6:] == '</OME>' or d[-10:] == "</OME:OME>")

        @property
        def is_ome(self):
            """Page contains OME-XML in ImageDescription tag."""
            if self.index > 1 or not self.description:
                return False
            d = self.description
            return re.match(r"<\?xml version *=", d[:20]) is not None and \
                re.match(r".*</(OME:)?OME>[ \n]*$", d[-20:], re.DOTALL) is not None

    TiffFile.report_func = lambda x: 0
    tifffile.tifffile.TiffPage = MyTiffPage

    if packaging.version.parse(tifffile.__version__) <= packaging.version.parse("2019.7.26"):
        asbool = tifffile.tifffile.asbool

        def _xml2dict(xml, sanitize=True, prefix=None):
            """Return XML as dict.

            >>> _xml2dict('<?xml version="1.0" ?><root attr="name"><key>1</key></root>')
            {'root': {'key': 1, 'attr': 'name'}}

            """
            from xml.etree import cElementTree as etree  # delayed import

            at = tx = ''
            if prefix:
                at, tx = prefix

            def astype(value):
                # return value as int, float, bool, or str
                if not isinstance(value, str):
                    return value

                for t in (int, float, asbool):
                    try:
                        return t(value)
                    except Exception:
                        pass
                return value

            def etree2dict(t):
                # adapted from https://stackoverflow.com/a/10077069/453463
                key = t.tag
                if sanitize:
                    key = key.rsplit('}', 1)[-1]
                d = {key: {} if t.attrib else None}
                children = list(t)
                if children:
                    dd = collections.defaultdict(list)
                    for dc in map(etree2dict, children):
                        for k, v in dc.items():
                            dd[k].append(astype(v))
                    d = {key: {k: astype(v[0]) if len(v) == 1 else astype(v)
                               for k, v in dd.items()}}
                if t.attrib:
                    d[key].update((at + k, astype(v)) for k, v in t.attrib.items())
                if t.text:
                    text = t.text.strip()
                    if children or t.attrib:
                        if text:
                            d[key][tx + 'value'] = astype(text)
                    else:
                        d[key] = astype(text)
                return d

            return etree2dict(etree.fromstring(xml))

        tifffile.xml2dict = _xml2dict
        tifffile.tifffile.xml2dict = _xml2dict
        import czifile
        czifile.czifile.xml2dict = _xml2dict

    if tifffile.__version__ == '0.15.1':
        import warnings
        import numpy
        from tifffile import product, squeeze_axes
        # noinspection PyProtectedMember
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
                # noinspection PyBroadException
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
                                    # noinspection PyUnresolvedReferences
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
                    # dtype = attr.get('PixelType', None)
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
