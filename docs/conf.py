# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

from datetime import date

import PartSeg

# -- Project information -----------------------------------------------------

project = "PartSeg"
copyright = f"{date.today().year}, PartSeg team"
author = "Grzegorz Bokota (LFSG)"

# The full version, including alpha/beta/rc tags

release = PartSeg.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_qt_documentation",
    "sphinx.ext.viewcode",
    "sphinx.ext.graphviz",
    "sphinx_autodoc_typehints",
    "PartSegCore.sphinx.auto_parameters",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx.ext.autosectionlabel",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "nature"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

master_doc = "index"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "PyQt5": ("https://www.riverbankcomputing.com/static/Docs/PyQt5", None),
    "Numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "packaging": ("https://packaging.pypa.io/en/latest/", None),
}

qt_documentation = "Qt5"

latex_engine = "xelatex"
