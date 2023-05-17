# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-15 15:32:24
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-05-16 18:24:37


import datetime
import os
import warnings
from collections import UserDict

import numpy as np
import pandas as pd

from .. import core as nap

# from treelib import Node, Tree
from rich.tree import Tree
from rich.panel import Panel
from rich import print

import json

def find_files(path, extension=".npz"):
    files = {}
    for f in os.scandir(path):
        if f.is_file() and f.name.endswith(extension):
            if extension == "npz":
                files[os.path.splitext(os.path.basename(f.path))[0]] = NPZFile(f.path)
            elif extension == "nwb":
                files[os.path.splitext(os.path.basename(f.path))[0]] = os.path.basename(f.path)
    return files

class _Container(object):
    """
        THe base container class that can be inherited by the other classes 
        to make sure the folder exists and other errors
    """
    pass

class NPZFile(object):

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.file = np.load(self.path)
        self.type = ""
        if 'index' in self.file.keys():
            self.type = "TsGroup"          
        elif 'columns' in self.file.keys():
            self.type = "TsdFrame"
        elif 'd' in self.file.keys():
            self.type = "Tsd"
        elif 't' in self.file.keys():
            self.type = "Ts"
        else:
            self.type = "IntervalSet"

    def load(self):        
        time_support = nap.IntervalSet(self.file['start'], self.file['end'])
        if 'index' in self.file.keys():            
            tsd = nap.Tsd(t=self.file['t'], d=self.file['index'], time_support = time_support)
            tsgroup = tsd.to_tsgroup()
            tsgroup.set_info(group = self.file['group'], location = self.file['location'])
            return tsgroup
        elif 'columns' in self.file.keys():
            return nap.TsdFrame(t=self.file['t'], d=self.file['d'], time_support=time_support, columns=self.file['columns'])
        elif 'd' in self.file.keys():
            return nap.Tsd(t=self.file['t'], d=self.file['d'], time_support=time_support)
        elif 't' in self.file.keys():
            return nap.Ts(t=self.file['t'], time_support=time_support)
        else:
            return time_support

    def save(self):
        pass


class Folder(UserDict):

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)

        self.npz_files = find_files(path, "npz")
        self.nwb_files = find_files(path, "nwb")

        self.data = {**self.npz_files, **self.nwb_files}
        
        UserDict.__init__(self, self.data)

    def __getitem__(self, key):
        if key.__hash__:
            if self.__contains__(key):
                if isinstance(self.data[key], NPZFile):
                    return self.data[key].load()
                else:
                    return self.data[key]
            else:
                raise KeyError("Can't find key {} in group index.".format(key))
        # else:            
        #     if isinstance(self.data[key], NPZFile):
        #         return self.data[key].load()
        #     else:
        #         return self.data[key]


    def __repr__(self):

        # tree = Tree()
        # tree.create_node("Project name : {}".format(self.name), "root")
        # for sub in self.subjects.keys():
        #     tree.create_node(sub, sub, parent = "root")
        #     for ses in self.subjects[sub].sessions.keys():
        #         tree.create_node(ses, sub+"_"+ses, parent = sub)
        # tree.show()

        tree = Tree(
            ":open_file_folder: {}".format(self.name),
            guide_style="bright_blue"
            )
        
        # NPZ files
        for file in self.npz_files.values():
            tree.add("[green]"+file.name+" \t|\t "+file.type)

        # NWB files
        for file in self.nwb_files.values():
            tree.add("[magenta]"+file+" \t|\t NWB file")

        print(tree)

        return ""

    def doc(self, name):
        # Search for json first
        json_filename = os.path.join(self.path, name + ".json")
        if os.path.isfile(json_filename):
            with open(json_filename, 'r') as ff:
                metadata = json.load(ff)
                text = "\n".join([" : ".join(it) for it in metadata.items()])
            panel = Panel.fit(text, border_style="green", title = os.path.join(self.path, name + ".npz"))
            print(panel)
        else:
            panel = Panel.fit("No metadata", border_style="red", title = os.path.join(self.path, name + ".npz"))
            print(panel)
        
        return ""


class Session(UserDict):

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)

        folders = [ f.path for f in os.scandir(path) if f.is_dir() ]

        self.folders = {}
        for s in folders:
            f = os.path.basename(s)
            self.folders[f] = Folder(s)

        self.npz_files = find_files(path, "npz")
        self.nwb_files = find_files(path, "nwb")

        self.data = {**self.folders, **self.npz_files, **self.nwb_files}
        
        UserDict.__init__(self, self.data)

    def __getitem__(self, key):
        if key.__hash__:
            if self.__contains__(key):
                if isinstance(self.data[key], NPZFile):
                    return self.data[key].load()
                else:
                    return self.data[key]
            else:
                raise KeyError("Can't find key {} in group index.".format(key))
        else:            
            if isinstance(self.data[key], NPZFile):
                return self.data[key].load()
            else:
                return self.data[key]

    def __repr__(self):

        # tree = Tree()
        # tree.create_node("Project name : {}".format(self.name), "root")
        # for sub in self.subjects.keys():
        #     tree.create_node(sub, sub, parent = "root")
        #     for ses in self.subjects[sub].sessions.keys():
        #         tree.create_node(ses, sub+"_"+ses, parent = sub)
        # tree.show()

        tree = Tree(
            ":open_file_folder: {}".format(self.name),
            guide_style="bright_blue"
            )
        

        # Folder
        for fold in self.folders.keys():
            tree.add(":open_file_folder: "+fold)

            # NPZ files
            for file in self.folders[fold].npz_files.values():
                tree.children[-1].add("[green]"+file.name+" \t|\t "+file.type)

            # NWB files
            for file in self.folders[fold].nwb_files.values():
                tree.children[-1].add("[magenta]"+file+" \t|\t NWB file")

        # NPZ files
        for file in self.npz_files.values():
            tree.add("[green]"+file.name+" \t|\t "+file.type)

        # NWB files
        for file in self.nwb_files.values():
            tree.add("[magenta]"+file+" \t|\t NWB file")

        print(tree)

        return ""      

    def save(self, obj, name, description=""):
        filepath = os.path.join(self.path, name)
        obj.save(filepath)        
        self.npz_files[name] = NPZFile(filepath+'.npz')
        self.data[name] = self.npz_files[name]
        
        metadata = {
            'time': 'today', 
            'info': description
            }

        with open(os.path.join(self.path, name+".json"), 'w') as ff:
            json.dump(metadata, ff)

    def doc(self, name):
        # Search for json first
        json_filename = os.path.join(self.path, name + ".json")
        if os.path.isfile(json_filename):
            with open(json_filename, 'r') as ff:
                metadata = json.load(ff)
                text = "\n".join([" : ".join(it) for it in metadata.items()])
            panel = Panel.fit(text, border_style="green", title = os.path.join(self.path, name + ".npz"))
            print(panel)
        else:
            panel = Panel.fit("No metadata", border_style="red", title = os.path.join(self.path, name + ".npz"))
            print(panel)
        
        return ""


class Subject(UserDict):

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        
        sessions = [ f.path for f in os.scandir(path) if f.is_dir() ]

        self.sessions = {}

        for s in sessions:
            ses = os.path.basename(s)
            self.sessions[ses] = Session(s)

        UserDict.__init__(self, self.sessions)

    def __getitem__(self, key):
        if key.__hash__:
            if self.__contains__(key):
                return self.data[key]
            else:
                raise KeyError("Can't find key {} in group index.".format(key))
        else:            
            return self.sessions[key]

    def __repr__(self):

        tree = Tree(
            ":open_file_folder: {}".format(self.name),
            guide_style="bright_blue"
            )
        
        # Session
        for ses in self.sessions.keys():
            tree.add(":open_file_folder: [blue]"+ses)

            # Folder
            for fold in self.sessions[ses].folders.keys():
                tree.children[-1].add(":open_file_folder: "+fold)

                # NPZ files
                for file in self.sessions[ses].folders[fold].npz_files.values():
                    tree.children[-1].children[-1].add("[green]"+file.name+" \t|\t "+file.type)

                # NWB files
                for file in self.sessions[ses].folders[fold].nwb_files.values():
                    tree.children[-1].children[-1].add("[magenta]"+file+" \t|\t NWB file")

            # NPZ files
            for file in self.sessions[ses].npz_files.values():
                tree.children[-1].add("[green]"+file.name+" \t|\t "+file.type)

            # NWB files
            for file in self.sessions[ses].nwb_files.values():
                tree.children[-1].add("[magenta]"+file+" \t|\t NWB file")

        print(tree)

        return ""            


class Project(UserDict):

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)

        subjects = [ f.path for f in os.scandir(path) if f.is_dir() ]

        self.subjects = {}

        for s in subjects:
            sub = os.path.basename(s)
            self.subjects[sub] = Subject(s)

        UserDict.__init__(self, self.subjects)

    def __getitem__(self, key):
        if key.__hash__:
            if self.__contains__(key):
                return self.data[key]
            else:
                raise KeyError("Can't find key {} in group index.".format(key))
        else:            
            return self.subjects[key]

    def __repr__(self):

        # tree = Tree()
        # tree.create_node("Project name : {}".format(self.name), "root")
        # for sub in self.subjects.keys():
        #     tree.create_node(sub, sub, parent = "root")
        #     for ses in self.subjects[sub].sessions.keys():
        #         tree.create_node(ses, sub+"_"+ses, parent = sub)
        # tree.show()

        tree = Tree(
            ":open_file_folder: Project name : {}".format(self.name),
            guide_style="bright_blue"
            )
        
        # Subject
        for sub in self.subjects.keys():
            tree.add(":open_file_folder:"+sub)

            # Session
            for ses in self.subjects[sub].sessions.keys():
                tree.children[-1].add(":open_file_folder: [blue]"+ses)

                # Folder
                for fold in self.subjects[sub].sessions[ses].folders.keys():
                    tree.children[-1].children[-1].add(":open_file_folder: "+fold)

                    # NPZ files
                    for file in self.subjects[sub].sessions[ses].folders[fold].npz_files.values():
                        tree.children[-1].children[-1].children[-1].add("[green]"+file.name+" \t|\t "+file.type)

                    # NWB files
                    for file in self.subjects[sub].sessions[ses].folders[fold].nwb_files.values():
                        tree.children[-1].children[-1].children[-1].add("[magenta]"+file+" \t|\t NWB file")

                # NPZ files
                for file in self.subjects[sub].sessions[ses].npz_files.values():
                    tree.children[-1].children[-1].add("[green]"+file.name+" \t|\t "+file.type)

                # NWB files
                for file in self.subjects[sub].sessions[ses].nwb_files.values():
                    tree.children[-1].children[-1].add("[magenta]"+file+" \t|\t NWB file")

        print(tree)

        return ""

    def __str__(self):
        return self.__repr__()

    def keys(self):
        return list(self.subjects.keys())