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

sys.path.insert(0, os.path.abspath('..'))


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
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',  # Links to source code
    'sphinxcontrib.apidoc',
    'myst_parser',  # Markdown support
    'sphinx_copybutton',  # Adds copy button to code blocks
    'sphinx_design',  # For layout components
]


templates_path = ['_templates']


# The Root document
root_doc = "index"


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'docstrings', 'nextgen', 'Thumbs.db', '.DS_Store']


# The reST default role (used for this markup: `text`) to use for all documents.
default_role = 'literal'

# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = False

# ----------------------------------------------------------------------------
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'

html_logo = "_static/Logo/Pynapple_final_logo.png"
html_favicon = "_static/Icon/Pynapple_final_icon.png"


# Additional theme options
# html_theme_options = {
#     "icon_links": [
#         {
#             "name": "GitHub",
#             "url": "https://github.com/pynapple-org/pynapple",
#             "icon": "fab fa-github",
#             "type": "fontawesome",
#         },
#         # {
#         #     "name": "StackOverflow",
#         #     "url": "https://stackoverflow.com/tags/seaborn",
#         #     "icon": "fab fa-stack-overflow",
#         #     "type": "fontawesome",
#         # },
#         {
#             "name": "Twitter",
#             "url": "https://twitter.com/thepynapple",
#             "icon": "fab fa-twitter",
#             "type": "fontawesome",
#         },
#     ],
#     "show_prev_next": False,
#     "navbar_start": ["navbar-logo"],
#     "navbar_end": ["navbar-icon-links"],
#     "header_links_before_dropdown": 8,
# }

html_context = {
    "default_mode": "light",
}


# ----------------------------------------------------------------------------
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

# -- Extension configuration -------------------------------------------------
apidoc_module_dir = "../pynapple"

















