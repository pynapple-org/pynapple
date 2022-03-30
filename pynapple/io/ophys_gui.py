# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-03-25 11:34:45
# @Last Modified by:   gviejo
# @Last Modified time: 2022-03-25 16:44:28

import sys, os, csv, getpass
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QStackedLayout  # add this import


def convert_to_dict(parent):
    childCount = parent.childCount()
    if not childCount:        
        return parent.text(1)
    values = {}
    for r in range(childCount):
        child = parent.child(r)
        values[child.text(0)] = convert_to_dict(child)
    return values


class OphysGUI(QMainWindow):

    def __init__(self, path=None):
        super().__init__()

        # Basic properties to return
        self.status = False
        self.path = path
        
        self.ophys = {}
        self.ophys = {
        	'device': {n:d for n, d in zip(
                ['name', 'description', 'manufacturer'],
                ['Microscope', '', '']
                )},
        	'OpticalChannel': {n:d for n,d in zip(
                ['name', 'description', 'emission_lambda'],
                ["OpticalChannel", "", "500."]
                )},
        	'ImagingPlane': {n:d for n,d in zip(
                ['name', 'imaging_rate', 'description', 'excitation_lambda', 'indicator', 'location'],
                ["ImagingPlane", "30.", "", "600.", "GCAMP", ""]
                )},
            'PlaneSegmentation': {n:d for n, d in zip(
                ['name', 'description'],
                ['PlaneSegmentation', '']
                )}
        }

        self.setWindowTitle("Calcium Imaging loader")
        self.setMinimumSize(640, 480)

        pagelayout = QVBoxLayout()

        # LOGO
        logo = QLabel("Pynapple")
        font = logo.font()
        font.setPointSize(20)
        logo.setFont(font)

        # # TREE VIEW
        self.tree = QTreeWidget()
        self.tree.setAlternatingRowColors( True )
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels([
            "Metadata",
            "Informations"
            ])
        self.tree.header().resizeSections(QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        
        items = []
        for key in self.ophys.keys():
            item = QTreeWidgetItem([key])
            for k, value in self.ophys[key].items():                
                child = QTreeWidgetItem([k, value])
                child.setFlags(child.flags() | Qt.ItemIsEditable)
                item.addChild(child)
            items.append(item)

        self.tree.insertTopLevelItems(0, items)
        self.tree.expandAll()

        # BOTTOM SAVING
        self.finalbuttons = QDialogButtonBox()
        self.finalbuttons.setOrientation(Qt.Horizontal)
        self.finalbuttons.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        self.finalbuttons.accepted.connect(self.accept)
        self.finalbuttons.rejected.connect(self.reject)

        pagelayout.addWidget(logo)
        pagelayout.addWidget(QLabel(path))

        pagelayout.addWidget(self.tree)

        pagelayout.addWidget(self.finalbuttons)

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

    def accept(self):
        self.status = True

        # Retrieve everything
        root = self.tree.invisibleRootItem()
        ophys_information = convert_to_dict(root)        
        self.ophys_information = ophys_information

        self.close()

    def reject(self):
        self.status = False
        self.close()





