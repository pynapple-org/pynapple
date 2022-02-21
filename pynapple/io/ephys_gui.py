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


class EphysGUI(QMainWindow):

    def __init__(self, path=None, groups = {}):
        super().__init__()

        # Basic properties to return
        self.status = False
        self.path = path
        self.groups = groups
        self.ephys = {}
        for k in groups.keys():
            self.ephys[k] = {}
            self.ephys[k]['electrodes'] = " ".join(groups[k].astype(np.str_))
            for n in ["name","description", "location", "device","position"]:
                self.ephys[k][n] = ''
            self.ephys[k]['device'] = {
                d:'' for d in ['name', 'description', 'manufacturer']
            }

        self.setWindowTitle("Ephys loader")
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
            "Groups",
            "Informations"
            ])
        self.tree.header().resizeSections(QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        
        items = []
        for key in self.ephys.keys():
            item = QTreeWidgetItem(['Group '+str(key)])
            for k, value in self.ephys[key].items():                
                if type(value) is str:
                    child = QTreeWidgetItem([k, value])
                    if k != 'electrodes':
                        child.setFlags(child.flags() | Qt.ItemIsEditable)
                elif type(value) is dict:
                    child = QTreeWidgetItem([k])
                    for d in self.ephys[key][k].keys():
                        child2 = QTreeWidgetItem([d, self.ephys[key][k][d]])
                        child2.setFlags(child.flags() | Qt.ItemIsEditable)
                        child.addChild(child2)

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
        ephys_information = convert_to_dict(root)        
        self.ephys_information = {int(k.split(' ')[1]):ephys_information[k] for k in ephys_information.keys()}

        self.close()

    def reject(self):
        self.status = False
        self.close()





