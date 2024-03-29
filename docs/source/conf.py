# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
import os
sys.path.insert(0, os.path.abspath("../.."))
import animal_soup


# -- Project information -----------------------------------------------------

project = 'animal-soup'
copyright = '2023, Caitlin Lewis, Kushal Kolar'
author = 'Caitlin Lewis, Kushal Kolar'
release = animal_soup.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.napoleon", 'sphinx.ext.autodoc', 'sphinx.ext.intersphinx']
autodoc_typehints = "description"
autodoc_mock_imports = ["torch"]

# templates_path = ['_templates']
exclude_patterns = []

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pytorch': ('https://pytorch.org/docs/stable/', None),
    'lightning': ('https://lightning.ai/docs/pytorch/stable/', None),
    'fastplotlib': ('https://fastplotlib.readthedocs.io/en/latest/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None)
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'
# html_theme_options = {"page_sidebar_items": ["class_page_toc"]}

autoclass_content = "both"

html_static_path = ['_static']
html_logo = "_static/logo.png"

autodoc_member_order = 'bysource'
