#!/usr/bin/env python

"""

@author: Guillaume Viejo
"""
# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-15 15:32:24
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-07-10 17:06:32

import json
import os
import string
from collections import UserDict
from datetime import datetime

from rich import print
from rich.panel import Panel

# from treelib import Node, Tree
from rich.tree import Tree

from .file import NPZFile, NWBFile


def _find_files(path, extension=".npz"):
    """Helper to locate files

    Parameters
    ----------
    path : TYPE
        Description
    extension : str, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    files = {}
    for f in os.scandir(path):
        if f.is_file() and f.name.endswith(extension):
            if extension == "npz":
                filename = os.path.splitext(os.path.basename(f.path))[0]
                filename.translate({ord(c): None for c in string.whitespace})
                files[filename] = NPZFile(f.path)
            elif extension == "nwb":
                filename = os.path.splitext(os.path.basename(f.path))[0]
                filename.translate({ord(c): None for c in string.whitespace})
                files[filename] = NWBFile(f.path)
    return files


def _walk_folder(tree, folder):
    """Summary

    Parameters
    ----------
    tree : TYPE
        Description
    folder : TYPE
        Description
    """
    # Folder
    for fold in folder.subfolds.keys():
        tree.add(":open_file_folder: " + fold)
        _walk_folder(tree.children[-1], folder.subfolds[fold])

    # NPZ files
    for file in folder.npz_files.values():
        tree.add("[green]" + file.name + " \t|\t " + file.type)

    # NWB files
    for file in folder.nwb_files.values():
        tree.add("[magenta]" + file.name + " \t|\t NWB file")


class Folder(UserDict):
    """
    Base class for all type of folders (i.e. Project, Subject, Sessions, ...).
    Handles files and sub-folders discovery

    Attributes
    ----------
    data : TYPE
        Description
    name : TYPE
        Description
    npz_files : TYPE
        Description
    nwb_files : TYPE
        Description
    path : TYPE
        Description
    subfolds : dict
        Description

    """

    def __init__(self, path, exclude=(), max_depth=4):
        """Summary

        Parameters
        ----------
        path : TYPE
            Description
        exclude : tuple, optional
            Description
        max_depth : int, optional
            Description
        """
        self.path = path
        self.name = os.path.basename(path)
        self._basic_view = Tree(
            ":open_file_folder: {}".format(self.name), guide_style="blue"
        )
        self._full_view = None

        # Search sub-folders
        subfolds = [f.path for f in os.scandir(path) if f.is_dir()]
        subfolds.sort()

        self.subfolds = {}

        for s in subfolds:
            sub = os.path.basename(s)
            self.subfolds[sub] = Folder(s)
            self._basic_view.add(":open_file_folder: [blue]" + sub)

        # Search files
        self.npz_files = _find_files(path, "npz")
        self.nwb_files = _find_files(path, "nwb")

        for filename, file in self.npz_files.items():
            self._basic_view.add("[green]" + file.name + " \t|\t " + file.type)

        for file in self.nwb_files.values():
            self._basic_view.add("[magenta]" + file.name + " \t|\t NWB file")

        # Putting everything together
        self.data = {**self.npz_files, **self.nwb_files, **self.subfolds}

        UserDict.__init__(self, self.data)

    def __str__(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.__repr__()

    def __repr__(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        print(self._basic_view)
        return ""

    def __getitem__(self, key):
        """Summary

        Parameters
        ----------
        key : TYPE
            Description

        Returns
        -------
        TYPE
            Description

        Raises
        ------
        KeyError
            Description
        """
        if key.__hash__:
            if self.__contains__(key):
                if isinstance(self.data[key], NPZFile):
                    data = self.data[key].load()
                    self.data[key] = data
                    # setattr(self, key, data)
                    return data
                elif isinstance(self.data[key], NWBFile):
                    return self.data[key].load()
                else:
                    return self.data[key]
            else:
                raise KeyError("Can't find key {} in group index.".format(key))

    # # # Gets called when an attribute is accessed
    # def __getattribute__(self, item):
    #     value = super(Folder, self).__getattribute__(item)

    #     if isinstance(value, NPZFile):
    #         data = value.load()
    #         setattr(self, item, data)
    #         self.data[item] = data
    #         return data
    #     else:
    #         return value

    def expand(self):
        """Display the full tree view. Equivalent to Folder.view

        Returns
        -------
        TYPE
            Description
        """
        if isinstance(self._full_view, Tree):
            print(self._full_view)
        else:
            tree = Tree(":open_file_folder: {}".format(self.name), guide_style="blue")

            # Folder
            for fold in self.subfolds.keys():
                tree.add(":open_file_folder: " + fold)
                _walk_folder(tree.children[-1], self.subfolds[fold])

            # NPZ files
            for file in self.npz_files.values():
                tree.add("[green]" + file.name + " \t|\t " + file.type)

            # NWB files
            for file in self.nwb_files.values():
                tree.add("[magenta]" + file.name + " \t|\t NWB file")

            self._full_view = tree

            print(tree)
        return None

    @property
    def view(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.expand()

    def save(self, name, obj, description=""):
        """Save a pynapple object in the folder in a single file in uncompressed ``.npz`` format.
        By default, the save function overwrite previously save file with the same name.

        Parameters
        ----------
        name : str
            Filename
        obj : Ts, Tsd, TsdFrame, TsGroup or IntervalSet
            Pynapple object.
        description : str, optional
            Metainformation added as a json sidecar.
        """
        filepath = os.path.join(self.path, name)
        obj.save(filepath)
        self.npz_files[name] = NPZFile(filepath + ".npz")
        self.data[name] = obj

        metadata = {"time": str(datetime.now()), "info": str(description)}

        with open(os.path.join(self.path, name + ".json"), "w") as ff:
            json.dump(metadata, ff, indent=2)

    def load(self):
        """Load all compatible NPZ files."""
        for k in self.npz_files.keys():
            self[k] = self.npz_files[k].load()

    # def add_metadata(self):
    #     """Summary"""
    #     pass

    def info(self, name):
        """Display the metadata within the json sidecar of a NPZ file

        Parameters
        ----------
        name : str
            Name of the npz file
        """
        self.metadata(name)

    def doc(self, name):
        """Display the metadata within the json sidecar of a NPZ file

        Parameters
        ----------
        name : str
            Name of the npz file
        """
        self.metadata(name)

    def metadata(self, name):
        """Display the metadata within the json sidecar of a NPZ file

        Parameters
        ----------
        name : str
            Name of the npz file
        """
        # Search for json first
        json_filename = os.path.join(self.path, name + ".json")
        if os.path.isfile(json_filename):
            with open(json_filename, "r") as ff:
                metadata = json.load(ff)
                text = "\n".join([" : ".join(it) for it in metadata.items()])
            panel = Panel.fit(
                text, border_style="green", title=os.path.join(self.path, name + ".npz")
            )
            print(panel)
        else:
            panel = Panel.fit(
                "No metadata",
                border_style="red",
                title=os.path.join(self.path, name + ".npz"),
            )
            print(panel)

        return None

    # def apply(self):
    #     """Summary"""
    #     pass
