# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../pygem/'))

project = 'pygem'
copyright = '2023, David Rounce'
author = 'David Rounce'
release = '0.2.5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

#templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

html_theme_options = {
    "repository_url": "https://github.com/drounce/PyGEM",
    "use_repository_button": True,
#    "use_issues_button": True,
#    "use_edit_page_button": True,
#    "path_to_docs": "docs",
#    # "home_page_in_toc": True,
#    "toc_title": 'On this page',
    }