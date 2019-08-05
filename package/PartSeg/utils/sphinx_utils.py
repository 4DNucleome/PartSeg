"""
This module contains sphinx extension supporting for build PartSeg documentation.

Currently only support of linking to Qt documentation.
"""
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from docutils.nodes import Element, TextElement
from docutils import nodes
from typing import List, Optional, Dict, Any
from sphinx.locale import _

from sphinx.ext.intersphinx import InventoryAdapter


# TODO add response to
#  https://stackoverflow.com/questions/47102004/how-to-properly-link-to-pyqt5-documentation-using-intersphinx

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
    if target.startswith("PyQt5"):
        obj_type_name = "sip:{}".format(node.get("reftype"))
        if target in inventories.main_inventory[obj_type_name]:
            proj, version, uri, dispname = inventories.main_inventory[obj_type_name][target]
            html_name = uri.split("/")[-1]
            uri = "https://doc.qt.io/qt-5/" + html_name
            # remove this line if you would like straight to pyqt documentation
            if version:
                reftitle = _('(in %s v%s)') % (proj, version)
            else:
                reftitle = _('(in %s)') % (proj,)
            newnode = nodes.reference('', '', internal=False, refuri=uri, reftitle=reftitle)
            if node.get('refexplicit'):
                # use whatever title was given
                newnode.append(contnode)
            else:
                # else use the given display name (used for :ref:)
                newnode.append(contnode.__class__(dispname, dispname))
            return newnode
    # print(node)  # print nodes with unresolved references
    return None


def setup(app: Sphinx) -> Dict[str, Any]:
    app.connect('missing-reference', missing_reference)
    return {
        'version': "0.9",
        'env_version': 1,
        'parallel_read_safe': True
    }
