import collections
import re

import packaging.version
import tifffile.tifffile
from tifffile import TiffFile, TiffPage

if tifffile.tifffile.TiffPage.__module__ != "PartSegImage.tifffile_fixes":  # noqa C901

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
        def is_ome(self):
            """Page contains OME-XML in ImageDescription tag."""
            if self.index > 1 or not self.description:
                return False
            d = self.description
            return (
                re.match(r"<\?xml version *=", d[:20]) is not None
                and re.match(r".*</(OME:)?OME>[ \n]*$", d[-20:], re.DOTALL) is not None
            )

    TiffFile.report_func = lambda x: 0
    tifffile.tifffile.TiffPage = MyTiffPage

    if packaging.version.parse(tifffile.__version__) <= packaging.version.parse("2019.7.26"):  # pragma: no cover
        asbool = tifffile.tifffile.asbool

        def _xml2dict(xml, sanitize=True, prefix=None):
            """Return XML as dict.

            >>> _xml2dict('<?xml version="1.0" ?><root attr="name"><key>1</key></root>')
            {'root': {'key': 1, 'attr': 'name'}}

            """
            from defusedxml import cElementTree as etree  # delayed import

            at = tx = ""
            if prefix:
                at, tx = prefix

            def astype(value):
                # return value as int, float, bool, or str
                if not isinstance(value, str):
                    return value

                for t in (int, float, asbool):
                    try:
                        return t(value)
                    except (TypeError, ValueError):
                        pass
                return value

            def etree2dict(t):
                # adapted from https://stackoverflow.com/a/10077069/453463
                key = t.tag
                if sanitize:
                    key = key.rsplit("}", 1)[-1]
                d = {key: {} if t.attrib else None}
                children = list(t)
                if children:
                    dd = collections.defaultdict(list)
                    for dc in map(etree2dict, children):
                        for k, v in dc.items():
                            dd[k].append(astype(v))
                    d = {key: {k: astype(v[0]) if len(v) == 1 else astype(v) for k, v in dd.items()}}
                if t.attrib:
                    d[key].update((at + k, astype(v)) for k, v in t.attrib.items())
                if t.text:
                    text = t.text.strip()
                    if children or t.attrib:
                        if text:
                            d[key][tx + "value"] = astype(text)
                    else:
                        d[key] = astype(text)
                return d

            return etree2dict(etree.fromstring(xml))

        tifffile.xml2dict = _xml2dict
        tifffile.tifffile.xml2dict = _xml2dict
        import czifile

        czifile.czifile.xml2dict = _xml2dict
