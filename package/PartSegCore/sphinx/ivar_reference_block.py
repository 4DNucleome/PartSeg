from typing import Dict, List, Tuple

from docutils import nodes
from docutils.nodes import Node
from sphinx.util.docfields import TypedField
from sphinx import addnodes


def patched_make_field(self, types: Dict[str, List[Node]], domain: str,
               items: Tuple, env: "BuildEnvironment" = None) -> nodes.field:
    def handle_item(fieldarg: str, content: str) -> nodes.paragraph:
        par = nodes.paragraph()
        par.extend(self.make_xrefs(self.rolename, domain, fieldarg,
                                   addnodes.literal_strong, env=env))
        if fieldarg in types:
            par += nodes.Text(' (')
            # NOTE: using .pop() here to prevent a single type node to be
            # inserted twice into the doctree, which leads to
            # inconsistencies later when references are resolved
            fieldtype = types.pop(fieldarg)
            if len(fieldtype) == 1 and isinstance(fieldtype[0], nodes.Text):
                typename = fieldtype[0].astext()
                par.extend(self.make_xrefs(self.typerolename, domain, typename,
                                           addnodes.literal_emphasis, env=env))
            else:
                par += fieldtype
            par += nodes.Text(')')
        par += nodes.Text(' -- ')
        par += content
        return par

    fieldname = nodes.field_name('', self.label)
    if len(items) == 1 and self.can_collapse:
        fieldarg, content = items[0]
        bodynode = handle_item(fieldarg, content)  # type: nodes.Node
    else:
        bodynode = self.list_type()
        for fieldarg, content in items:
            bodynode += nodes.list_item('', handle_item(fieldarg, content))
    fieldbody = nodes.field_body('', bodynode)
    return nodes.field('', fieldname, fieldbody)


TypedField.make_field = patched_make_field