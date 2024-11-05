"""
The Folder class helps to navigate a hierarchical data tree.
"""

import json
import string
from collections import UserDict
from datetime import datetime
from pathlib import Path

from rich.console import Console  # , ConsoleOptions, RenderResult
from rich.panel import Panel
from rich.tree import Tree

from .interface_npz import NPZFile
from .interface_nwb import NWBFile


def _find_files(path, extension=".npz"):
    """Helper to locate files

    Parameters
    ----------
    path : str or Path
        The directory path where files will be searched.
    extension : str, optional
        The file extension to look for, default is ".npz".

    Returns
    -------
    dict
        Dictionary with filenames (without extension and whitespace) as keys
        and NPZFile or NWBFile objects as values.
    """
    extension = extension if extension.startswith(".") else "." + extension
    path = Path(path)  # Ensure path is a Path object
    files = {}
    extensions_dict = {".npz": NPZFile, ".nwb": NWBFile}
    assert extension in extensions_dict.keys(), f"Extension {extension} not supported"

    for f in path.iterdir():
        if f.is_file() and f.suffix == extension:
            filename = f.stem
            filename = filename.translate({ord(c): None for c in string.whitespace})
            files[filename] = extensions_dict[extension](f)

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
    Dictionnary like object to walk and loop through nested folders.
    Handles files and sub-folders discovery

    Attributes
    ----------
    data : dict
        Dictionary holidng all the pynapple objects found in the folder.
    name : str
        Name of the folder
    npz_files : list
        List of npz files found in the folder
    nwb_files : list
        List of nwb files found in the folder
    path : str
        Absolute path of the folder
    subfolds : dict
        Dictionary of all the subfolders

    """

    def __init__(self, path):  # , exclude=(), max_depth=4):
        """Initialize the Folder object

        Parameters
        ----------
        path : str
            Path to the folder
        """
        path = Path(path)
        self.path = path
        self.name = self.path.name
        self._basic_view = Tree(
            ":open_file_folder: {}".format(self.name), guide_style="blue"
        )
        self._full_view = None

        # Search sub-folders
        subfolds = [
            p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")
        ]

        subfolds.sort()

        self.subfolds = {}

        for s in subfolds:
            sub = s.name
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
        """View of the object"""
        with Console() as console:
            console.print(self._basic_view)
        return ""

    # def __repr__(self):
    #     """View of the object"""
    #     print(self._basic_view)

    def __getitem__(self, key):
        """Get subfolder or load file.

        Parameters
        ----------
        key : str

        Returns
        -------
        (Ts, Tsd, TsdFrame, TsGroup, IntervalSet, Folder or NWBFile)

        Raises
        ------
        KeyError
            If key is not in the dictionary
        """
        if key.__hash__:
            if self.__contains__(key):
                if isinstance(self.data[key], NPZFile):
                    data = self.data[key].load()
                    self.data[key] = data
                    # setattr(self, key, data)
                    return data
                elif isinstance(self.data[key], NWBFile):
                    return self.data[key]
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

    def _generate_tree_view(self):
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

    def expand(self):
        """Display the full tree view. Equivalent to Folder.view"""
        if not isinstance(self._full_view, Tree):
            self._generate_tree_view()

        with Console() as console:
            console.print(self._full_view)

        return None

    @property
    def view(self):
        """Summary"""
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
        filepath = self.path / (name + ".npz")
        obj.save(filepath)
        self.npz_files[name] = NPZFile(filepath)
        self.data[name] = obj

        metadata = {"time": str(datetime.now()), "info": str(description)}

        with open(self.path / (name + ".json"), "w") as ff:
            json.dump(metadata, ff, indent=2)

        # regenerate the tree view
        self._generate_tree_view()

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
        json_filename = self.path / (name + ".json")
        title = self.path / (name + ".npz")
        if json_filename.exists():
            with open(json_filename, "r") as ff:
                metadata = json.load(ff)
                text = "\n".join([" : ".join(it) for it in metadata.items()])
            panel = Panel.fit(text, border_style="green", title=str(title))
        else:
            panel = Panel.fit(
                "No metadata",
                border_style="red",
                title=str(title),
            )
        with Console() as console:
            console.print(panel)

        return None

    # def apply(self):
    #     """Summary"""
    #     pass
