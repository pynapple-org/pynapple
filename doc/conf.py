# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import time
import pynapple
from pathlib import Path

sys.path.insert(0, str(Path('..', '.').resolve()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pynapple'
copyright = f'2021-{time.strftime("%Y")}, Guillaume Viejo'
author = 'Guillaume Viejo, Edoardo Balzani'
version = release = pynapple.__version__


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',  # Links to source code
    'myst_parser',  # Markdown support
    'sphinx_copybutton',  # Adds copy button to code blocks
    'sphinx_design',  # For layout components
]


templates_path = ['_templates']


exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
# Additional theme options
html_theme_options = {
    "navbar_end": ["search-field.html", "navbar-icon-links"],
    # "primary_sidebar_end": ["custom-template.html"],
    # "use_edit_page_button": True,  # Allows users to edit the docs via GitHub
    # "external_links": [
    #     {"name": "GitHub", "url": "https://github.com/your-repo"},
    # ],
    # "icon_links": [
    #     {
    #         "name": "GitHub",
    #         "url": "https://github.com/your-repo",
    #         "icon": "fa-brands fa-github",
    #     },
    # ],
}


# -- Autodoc and Napoleon Options -------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
napoleon_numpy_docstring = True

# Path for static files (custom stylesheets or JavaScript)
html_static_path = ['_static']

# Copybutton settings (to hide prompt)
copybutton_prompt_text = r">>> |\$ |# "
copybutton_prompt_is_regexp = True

# Enable markdown and notebook support
myst_enable_extensions = ["colon_fence"]  # For improved markdown 



















