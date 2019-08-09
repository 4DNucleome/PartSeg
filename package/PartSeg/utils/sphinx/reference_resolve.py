"""
This module contains sphinx extension supporting for build PartSeg documentation.

this extensio provides one configuration option:

`qt_documentation` with possibe values:

 * PyQt - linking to PyQt documentation on https://www.riverbankcomputing.com/static/Docs/PyQt5/api/ (incomplete)
 * Qt - linking to Qt documentation on "https://doc.qt.io/qt-5/" (default)
 * PySide - linking to PySide documentation on  "https://doc.qt.io/qtforpython/PySide2/"
"""
import re
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from docutils.nodes import Element, TextElement
from docutils import nodes
from typing import List, Optional, Dict, Any
from sphinx.locale import _

from sphinx.ext.intersphinx import InventoryAdapter

from qtpy import QT_VERSION

# TODO add response to
#  https://stackoverflow.com/questions/47102004/how-to-properly-link-to-pyqt5-documentation-using-intersphinx

signal_slot_uri = {
    "Qt": "https://doc.qt.io/qt-5/signalsandslots.html",
    "PySide": "https://doc.qt.io/qtforpython/overviews/signalsandslots.html",
    "PyQt": "https://www.riverbankcomputing.com/static/Docs/PyQt5/signals_slots.html"
}

signal_name = {
    "Qt": "Signal",
    "PySide": "Signal",
    "PyQt": "pyqtSignal"
}

slot_name = {
    "Qt": "Slot",
    "PySide": "Slot",
    "PyQt": "pyqtSlot"
}

signal_pattern = re.compile(r'((\w+\d?\.QtCore\.)|(QtCore\.)|(\.)())?(pyqt)?Signal')
slot_pattern = re.compile(r'((\w+\d?\.QtCore\.)|(QtCore\.)|(\.)())?(pyqt)?Slot')

def missing_reference(app: Sphinx, env: BuildEnvironment, node: Element, contnode: TextElement
                      ) -> Optional[nodes.reference]:
    """Linking to Qt documentation."""
    target: str = node['reftarget']
    inventories = InventoryAdapter(env)
    objtypes = None  # type: Optional[List[str]]
    if node['reftype'] == 'any':
        # we search anything!
        objtypes = ['%s:%s' % (domain.name, objtype)
                    for domain in env.domains.values()
                    for objtype in domain.object_types]
        domain = None
    else:
        domain = node.get('refdomain')
        if not domain:
            # only objects in domains are in the inventory
            return None
        objtypes = env.get_domain(domain).objtypes_for_role(node['reftype'])
        if not objtypes:
            return None
        objtypes = ['%s:%s' % (domain, objtype) for objtype in objtypes]
    if target.startswith("PySide2"):
        head, tail = target.split(".", 1)
        target = "PyQt5." + tail
    obj_type_name = "sip:{}".format(node.get("reftype"))
    if obj_type_name not in inventories.named_inventory["PyQt"]:
        return None
    target_list = [target, "PyQt5." + target]
    target_list += [name + "." + target for name in inventories.named_inventory["PyQt"]["sip:module"].keys()]
    if signal_pattern.match(target):
        uri = signal_slot_uri[app.config.qt_documentation]
        dispname = signal_name[app.config.qt_documentation]
        version = QT_VERSION
    elif slot_pattern.match(target):
        uri = signal_slot_uri[app.config.qt_documentation]
        dispname = slot_name[app.config.qt_documentation]
        version = QT_VERSION
    else:
        for target_name in target_list:
            if target_name in inventories.main_inventory[obj_type_name]:
                proj, version, uri, dispname = inventories.named_inventory["PyQt"][obj_type_name][target_name]
                print(node)  # print nodes with unresolved references
                break
        else:
            return None
        if app.config.qt_documentation == "Qt":
            html_name = uri.split("/")[-1]
            uri = "https://doc.qt.io/qt-5/" + html_name
        elif app.config.qt_documentation == "PySide":
            html_name = "/".join(target.split(".")[1:]) + ".html"
            uri = "https://doc.qt.io/qtforpython/PySide2/" + html_name

    # remove this line if you would like straight to pyqt documentation
    if version:
        reftitle = _('(in %s v%s)') % (app.config.qt_documentation, version)
    else:
        reftitle = _('(in %s)') % (app.config.qt_documentation,)
    newnode = nodes.reference('', '', internal=False, refuri=uri, reftitle=reftitle)
    if node.get('refexplicit'):
        # use whatever title was given
        newnode.append(contnode)
    else:
        # else use the given display name (used for :ref:)
        newnode.append(contnode.__class__(dispname, dispname))
    return newnode


def setup(app: Sphinx) -> Dict[str, Any]:
    app.connect('missing-reference', missing_reference)
    app.add_config_value('qt_documentation', "Qt", True)
    return {
        'version': "0.9",
        'env_version': 1,
        'parallel_read_safe': True
    }
