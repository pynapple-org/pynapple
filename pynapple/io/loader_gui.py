# -*- coding: utf-8 -*-
import sys, os, csv
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QStackedLayout  # add this import


class EpochsTable(QTableWidget):
    def __init__(self, r, c, path):
        super().__init__(r, c)
        self.path = path
        self.setHorizontalHeaderLabels(['start', 'end', 'label'])        
        prelabel = QTableWidgetItem('sleep/wake')
        prelabel.setForeground(QColor('grey'))        
        self.setItem(0, 2, prelabel)

        self.check_change = True
        self.cellChanged.connect(self.c_current)

        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)     
        for i in range(3):  
            header.setSectionResizeMode(i, QHeaderView.Stretch)

        self.epochs = pd.DataFrame(index = [], columns = ['start', 'end', 'label'])
        self.show()

    def c_current(self):
        if self.check_change:
            row = self.currentRow()
            col = self.currentColumn()
            value = self.item(row, col)
            value = value.text()
            if col in [0,1]: value = float(value)
            self.epochs.loc[row,self.epochs.columns[col]] = value
            if row==0 and col==2: self.item(0,2).setForeground(QColor('black'))

    def open_sheet(self):
        self.check_change = False
        suggested_dir = self.path if self.path else os.getenv('HOME')
        path = QFileDialog.getOpenFileName(self, 'Open CSV', suggested_dir, 'CSV(*.csv)')
        if path[0] != '':
            self.epochs = pd.read_csv(path[0], usecols=[0,1], header = None, names = ['start', 'end'])
            self.setRowCount(0)
            for r in self.epochs.index:
                row = self.rowCount()
                self.insertRow(row)
                for column, clabel in enumerate(['start', 'end']):
                    item = QTableWidgetItem(str(self.epochs.loc[r,clabel]))
                    self.setItem(row, column, item)
            self.epochs['label'] = ''
        self.check_change = True

    def update_path_info(self, path):
        self.path = path

class EpochsTab(QWidget):
    def __init__(self, path, parent=None):
        super(EpochsTab, self).__init__(parent)

        self.path = path
        self.time_units = 's'

        self.layout = QVBoxLayout(self)

        # Select time units
        vbox = QHBoxLayout()
        vbox.addWidget(QLabel("Please select the time units :"))
        vbox.setAlignment(Qt.AlignLeft)
        for tu in ['s', 'ms', 'us']:
            tubox = QRadioButton(tu, self)
            tubox.toggled.connect(self._time_units)
            if tu == 's':
                tubox.setChecked(True)
            vbox.addWidget(tubox)

        self.layout.addLayout(vbox)        

        # Table view 
        self.table = EpochsTable(1, 3, self.path)
        self.layout.addWidget(self.table)  

        # Add row
        self.addRowBtn = QPushButton("Add row")
        self.addRowBtn.clicked.connect(self.add_row)
        self.layout.addWidget(self.addRowBtn)

        # Load a csv
        button = QPushButton("Load csv file")
        button.clicked.connect(self.load_csv_file)
        self.layout.addWidget(button)

        self.layout.addStretch()

    def load_csv_file(self, s):        
        self.table.open_sheet()

    def _time_units(self, s):
        rbtn = self.sender()
        if rbtn.isChecked() == True:
            self.time_units = rbtn.text()

    def add_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.update()

    def update_path_info(self, path):
        self.table.update_path_info(path)

class SessionInformationTab(QWidget):
    def __init__(self, path=None, parent=None):
        super(SessionInformationTab, self).__init__(parent)
        
        self.session_information = {
            'path':path,
            'name':None
        }
                
        self.layout = QVBoxLayout(self)

        # Session name
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Session name:"))
        hbox.setAlignment(Qt.AlignLeft)
        self.name = QLineEdit()
        self.name.textChanged.connect(self.update_name)
        hbox.addWidget(self.name)

        self.layout.addLayout(hbox)        

        # Table view 
        self.layout.addWidget(QLabel("Additional informations :"))

        self.table = QTableWidget(2,2)
        self.layout.addWidget(self.table)

        self.table.setHorizontalHeaderLabels(['key', 'value'])
        self.table.setItem(0, 0, QTableWidgetItem("experimentalist"))
        self.table.setItem(1, 0, QTableWidgetItem("genotype"))

        self.table.cellChanged.connect(self.c_current)

        header = self.table.horizontalHeader()
        for i in range(2): header.setSectionResizeMode(i, QHeaderView.Stretch)
        
        self.addRowBtn = QPushButton("Add row")
        self.addRowBtn.clicked.connect(self.add_row)
        self.layout.addWidget(self.addRowBtn)
        self.layout.addStretch()

        self.update_path_info(path)

    def update_path_info(self, path):
        self.session_information['path'] = path 
        self.session_information['name'] = os.path.basename(path) if path else None
        self.name.setText(self.session_information['name'])
        # print(self.session_information)

    def c_current(self):
        row = self.table.currentRow()
        col = self.table.currentColumn()
        value = self.table.item(row, col)
        value = value.text()
        key = self.table.item(row,0).text()
        self.session_information[key] = value
        # print(self.session_information)

    def add_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.update()

    def update_name(self, name):       
        self.name.setText(name)
        self.session_information['name'] = name
        # print(self.session_information)

class TrackingTab(QWidget):
    def __init__(self, parent=None):
        super(TrackingTab, self).__init__(parent)
        lay = QVBoxLayout(self)
        # Buttons
        button_start = QPushButton("start") #self.lang["btn_start"])
        button_stop = QPushButton("stop") #self.lang["btn_stop"])
        lay.addWidget(button_start)
        lay.addWidget(button_stop)
        # lay.addStretch()

class HelpTab(QWidget):
    def __init__(self, parent=None):
        super(HelpTab, self).__init__(parent)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Help"))


class BaseLoaderGUI(QMainWindow):

    def __init__(self, path=None):
        super().__init__()

        # Basic properties to return
        self.status = False
        self.path = path

        self.setWindowTitle("The minimalist session loader")
        self.setMinimumSize(600, 480)

        pagelayout = QVBoxLayout()

        # LOGO
        logo = QLabel("Pynapple")
        font = logo.font()
        font.setPointSize(20)
        logo.setFont(font)

        toplayout = QHBoxLayout()
        toplayout.addWidget(QLabel("Data directory:"))
        self.directory_line = QLineEdit(self.path)
        toplayout.addWidget(self.directory_line)
        fileselect = QPushButton("Browse")
        fileselect.clicked.connect(self.select_folder)
        toplayout.addWidget(fileselect)

        # TABS
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(False)

        self.tab_session = SessionInformationTab(self.path)
        self.tab_epoch = EpochsTab(self.path)
        self.tab_tracking = TrackingTab()
        self.tab_help = HelpTab()
        
        self.tabs.addTab(self.tab_session, 'Session Information')
        self.tabs.addTab(self.tab_epoch, 'Epochs')
        self.tabs.addTab(self.tab_tracking, 'Tracking')
        self.tabs.addTab(self.tab_help, 'Help')

        # BOTTOM SAVING
        self.finalbuttons = QDialogButtonBox()
        self.finalbuttons.setOrientation(Qt.Horizontal)
        self.finalbuttons.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        self.finalbuttons.accepted.connect(self.accept)
        self.finalbuttons.rejected.connect(self.reject)

        pagelayout.addWidget(logo)
        pagelayout.addLayout(toplayout)
        pagelayout.addWidget(self.tabs)
        pagelayout.addWidget(self.finalbuttons)

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

        self.show()

    def accept(self):
        self.status = True
        # Collect all the information acquired
        self.session_information = self.tab_session.session_information
        self.epochs = self.tab_epoch.table.epochs
        self.time_units_epochs = self.tab_epoch.time_units
        self.close()

    def reject(self):
        self.status = False
        self.close()

    def select_folder(self):
        self.path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.directory_line.setText(self.path)
        self.tab_session.update_path_info(self.path)
        self.tab_epoch.update_path_info(self.path)






