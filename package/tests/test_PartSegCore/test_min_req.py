import os
import re
from importlib.metadata import requires, version

import pytest

requires_list = [x for x in requires("PartSeg") if ";" not in x and ">=" in x]
requires_regexp = re.compile(r"^([A-Za-z][A-Za-z0-9\-]+).*>=(\d[^,)]+)")


@pytest.mark.skipif(not os.environ.get("MINIMAL_REQUIREMENTS", False), reason="not running minimum requirement test")
@pytest.mark.parametrize("requirement_specifier", requires_list)
def test_min_req(requirement_specifier):
    match = requires_regexp.match(requirement_specifier)
    assert match is not None
    package_name = match.group(1)
    version_specifier = match.group(2)
    assert version(package_name) == version_specifier
