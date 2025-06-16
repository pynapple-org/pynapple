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
from pathlib import Path

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("sphinxext"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pynapple"
copyright = f'2021-{time.strftime("%Y")}'
author = "Guillaume Viejo"
from importlib.metadata import version

release: str = version("pynapple")
# this will grab major.minor.patch (excluding any .devN afterwards, which should only
# show up when building locally during development)
version: str = ".".join(release.split(".")[:3])


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",  # Links to source code
    "sphinx.ext.doctest",
    "sphinx_copybutton",  # Adds copy button to code blocks
    "sphinx_design",  # For layout components
    "myst_nb",
    "sphinx_contributors",
    "matplotlib.sphinxext.plot_directive",
    # 'sphinxcontrib.apidoc'
    # 'sphinx_gallery.gen_gallery',
    # 'myst_sphinx_gallery',
]


templates_path = ["_templates"]


# The Root document
root_doc = "index"


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "docstrings", "nextgen", "Thumbs.db", ".DS_Store"]


# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = True
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

# apidoc_module_dir = '../pynapple'
# apidoc_output_dir = 'reference'
# apidoc_excluded_paths = ['tests']
# apidoc_separate_modules = True


# ----------------------------------------------------------------------------
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

html_logo = "_static/Logo/Pynapple_final_logo.png"
html_favicon = "_static/Icon/Pynapple_final_icon.png"


# Additional theme options
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/pynapple-org/pynapple",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/thepynapple",
            "icon": "fab fa-twitter",
            "type": "fontawesome",
        },
    ],
    "show_prev_next": True,
    "header_links_before_dropdown": 5,
}

html_context = {
    "default_mode": "light",
}

html_sidebars = {
    "index": [],
    "installing": [],
    "about": [],
    "releases": [],
    "external": [],
    "pynajax": [],
    "citing": [],
    "**": ["search-field.html", "sidebar-nav-bs.html"],
}

# # Path for static files (custom stylesheets or JavaScript)
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Copybutton settings (to hide prompt)
copybutton_prompt_text = r">>> |\$ |# "
copybutton_prompt_is_regexp = True

# Enable markdown and notebook support
myst_enable_extensions = ["colon_fence"]  # For improved markdown

# # ----------------------------------------------------------------------------
# # -- Autodoc and Napoleon Options -------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
napoleon_numpy_docstring = True


nitpicky = True

# Set timeout in seconds (e.g., 15 minutes)
nb_execution_timeout = 60 * 15

# Execute notebooks during the build:
nb_execution_mode = "cache"
nb_execution_raise_on_error = True
