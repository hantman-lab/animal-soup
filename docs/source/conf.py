# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# from bs4 import BeautifulSoup
# from typing import *
# import animal_soup

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'animal-soup'
copyright = '2023, Caitlin Lewis'
author = 'Caitlin Lewis'
release = "0.0.1a1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon"]
autodoc_typehints = "description"

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
#html_theme_options = {"page_sidebar_items": ["class_page_toc"]}
autoclass_content = "both"

html_static_path = ['_static']

autodoc_member_order = 'bysource'
