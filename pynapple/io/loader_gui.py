# -*- coding: utf-8 -*-
import sys, os, csv
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QStackedLayout  # add this import
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np

def check_ttl_detection(file, n_channels=1, channel=0, bytes_size=2, fs=20000.0):
    """
        load ttl from analogin.dat
    """
    f = open(file, 'rb')
    startoffile = f.seek(0, 0)
    endoffile = f.seek(0, 2)
    n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
    f.close()
    with open(file, 'rb') as f:
        data = np.fromfile(f, np.uint16).reshape((n_samples, n_channels))
    if n_channels == 1:
        data = data.flatten().astype(np.int32)
    else:
        data = data[:,channel].flatten().astype(np.int32)
    data = data/data.max()
    peaks,_ = scipy.signal.find_peaks(np.diff(data), height=0.5)
    timestep = np.arange(0, len(data))/fs
    analogin = pd.Series(index = timestep, data = data)
    peaks+=1
    ttl = pd.Series(index = timestep[peaks], data = data[peaks])    
    plt.figure()
    plt.plot(analogin)
    plt.plot(ttl, 'o')
    plt.title(file)
    plt.show()
    return


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
            'name':None,
            'description':None
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

        self.table = QTableWidget(3,2)
        self.layout.addWidget(self.table)

        self.table.setHorizontalHeaderLabels(['key', 'value'])
        self.table.setItem(0, 0, QTableWidgetItem("description"))
        self.table.setItem(1, 0, QTableWidgetItem("experimentalist"))
        self.table.setItem(2, 0, QTableWidgetItem("genotype"))

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
    def __init__(self, path=None, parent=None):
        super(TrackingTab, self).__init__(parent)
        self.path = path
        self.time_units = 's'
        self.tracking_method = 'Optitrack'
        self.csv_files = []
        self.parameters = pd.DataFrame(
            columns = [ 'csv', 
                        'ttl', 
                        'n_channels', 
                        'tracking_channel', 
                        'bytes_size',
                        'fs',
                        'epoch'])
        self.ttl_param_widgets = {}

        self.layout = QVBoxLayout(self)
        
        laytop = QHBoxLayout()
        laytop.addWidget(QLabel("Tracking system: "))
        combobox1 = QComboBox()
        combobox1.addItems(['Optitrack', 'Deep Lab Cut', 'Other'])
        combobox1.currentTextChanged.connect(self.get_tracking_method)
        laytop.addWidget(combobox1)
        
        # Load a csv
        button = QPushButton("Load csv file(s)")
        button.clicked.connect(self.load_csv_files)
        button.resize(button.sizeHint())
        laytop.addWidget(button)

        laytop.addStretch()

        self.layout.addLayout(laytop)

        # Table view
        self.table = QTableWidget(1,8)
        self.table.setHorizontalHeaderLabels(
            ['CSV file', 
            'TTL file', 
            'Number\nof\nchannels', 
            'Tracking\nchannel', 
            'Bytes\nsize', 
            'Sampling\nfrequency\n(Hz)', 
            'Start\nepoch',
            'TTL\ndetection'
            ]
            )
        self.table.itemDoubleClicked.connect(self.change_file)
        header = self.table.horizontalHeader()       
        header.setSectionResizeMode(QHeaderView.Stretch)

        self.layout.addWidget(self.table)

        self.layout.addStretch()

    def load_csv_files(self, s):
        suggested_dir = self.path if self.path else os.getenv('HOME')
        paths = QFileDialog.getOpenFileNames(self, 'Open CSV', suggested_dir, 'CSV(*.csv)')        
        self.csv_files = paths[0]
        self.table.setRowCount(len(self.csv_files))

        for i in range(len(self.csv_files)):
            path = os.path.dirname(self.csv_files[i])
            files = os.listdir(os.path.dirname(self.csv_files[i]))
            filename = os.path.basename(self.csv_files[i])

            # Updating parameters table
            self.parameters.loc[filename, 'csv'] = self.csv_files[i]            

            # Set filename in place
            self.table.setItem(i, 0, QTableWidgetItem(filename))

            # Infer the epoch
            n = os.path.splitext(filename)[0].split('_')[-1]
            nepoch = QSpinBox()
            if n.isdigit(): 
                nepoch.setValue(int(n))
                self.parameters.loc[filename,'epoch'] = int(n)
            nepoch.valueChanged.connect(self.change_ttl_params)
            self.table.setCellWidget(i, 6, nepoch)

            # Infer the ttl file
            possiblettlfile = [f for f in files if '_'+n in f and f != filename]
            if len(possiblettlfile):
                item = QTableWidgetItem(possiblettlfile[0])
                self.parameters.loc[filename,'ttl'] = os.path.join(path, possiblettlfile[0])
            else:
                item = QTableWidgetItem("Select ttl file")
            self.table.setItem(i, 1, item)
            item.setFlags( item.flags() ^ Qt.ItemIsEditable)

            # Default analogin parameters
            self.fill_default_value_parameters(filename, i)

            # Check button
            check_button = QPushButton("Check")
            check_button.clicked.connect(self.check_ttl)
            self.table.setCellWidget(i,7,check_button)

        header = self.table.horizontalHeader()       
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)

    def check_ttl(self):
        r = self.table.currentRow()
        check_ttl_detection(
            file=self.parameters.loc[self.parameters.index[r],'ttl'],
            n_channels=self.parameters.loc[self.parameters.index[r],'n_channels'],
            channel=self.parameters.loc[self.parameters.index[r],'tracking_channel'],
            bytes_size=self.parameters.loc[self.parameters.index[r],'bytes_size'],
            fs=self.parameters.loc[self.parameters.index[r],'fs']
            )
        
        
    def fill_default_value_parameters(self, filename, row):        
        ttl_param_widgets = {}
        iterates = zip(
            ['n_channels', 'tracking_channel', 'bytes_size', 'fs'],
            [2,3,4,5],
            [1, 0, 2, 20000.0]
            )        
        for key, col, dval in iterates:
            self.parameters.loc[filename,key] = dval
            if key == 'fs':
                ttl_param_widgets[key] = QDoubleSpinBox()
                ttl_param_widgets[key].setMaximum(100000.0)
                ttl_param_widgets[key].setSingleStep(1000.0)
            else:
                ttl_param_widgets[key] = QSpinBox()
            ttl_param_widgets[key].setValue(dval)
            ttl_param_widgets[key].valueChanged.connect(self.change_ttl_params)
            self.table.setCellWidget(row, col, ttl_param_widgets[key])
        self.ttl_param_widgets[row] = ttl_param_widgets

    def change_file(self, item):
        suggested_dir = self.path if self.path else os.getenv('HOME')
        if self.table.currentColumn() == 0:
            paths = QFileDialog.getOpenFileName(self, 'Open CSV', suggested_dir, 'CSV(*.csv)')        
        elif self.table.currentColumn() == 1:
            paths = QFileDialog.getOpenFileName(self, 'Open TTL file', suggested_dir)
        filename = os.path.basename(paths[0])
        item.setText(filename)
        self.parameters.iloc[self.table.currentRow(),self.table.currentColumn()] = paths[0]

    def get_tracking_method(self, s):
        self.tracking_method = s

    def change_ttl_params(self, value):
        if self.table.currentColumn() == 5: 
            self.parameters.iloc[self.table.currentRow(),self.table.currentColumn()] = float(value)
        else:
            self.parameters.iloc[self.table.currentRow(),self.table.currentColumn()] = int(value)    

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
        self.setMinimumSize(760, 480)

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
        self.tab_tracking = TrackingTab(self.path)
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
        self.tab_tracking.update_path_info(self.path)





