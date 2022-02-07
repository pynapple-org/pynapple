# -*- coding: utf-8 -*-
import sys, os, csv, getpass
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QStackedLayout  # add this import
import scipy.signal
import numpy as np
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg

class TTLDetection(QDialog):

    def __init__(self, parent, file, n_channels=1, channel=0, bytes_size=2, fs=20000.0):
        super(TTLDetection, self).__init__(parent)
        self.setWindowTitle("Check TTL detection")
        self.setMinimumSize(640, 480)
        self.threshold = 0.3
        self.status = False

        self.graphWidget = pg.PlotWidget()

        self.load_ttl(file, n_channels, channel, bytes_size, fs)
        self.plot_ttl()

        slider = QDoubleSpinBox()
        slider.setMinimum(0)
        slider.setMaximum(1)
        slider.setSingleStep(0.1)
        slider.lineEdit().setEnabled(False)        
        slider.setValue(self.threshold)
        slider.valueChanged.connect(self.change_threshold)
        slider.setGeometry(100, 100, 100, 40)

        layout = QVBoxLayout()        

        layout2 = QHBoxLayout()
        layout2.addWidget(QLabel("TTL threshold: "))
        layout2.addWidget(slider)
        layout2.addStretch()
        layout.addLayout(layout2)

        layout.addWidget(self.graphWidget)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
            
        self.plot_ttl()

        self.setLayout(layout)

    def load_ttl(self, file, n_channels, channel, bytes_size, fs):
        f = open(file, 'rb')
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
        f.close()
        data = np.memmap(file, np.uint16, 'r', shape = (n_samples, n_channels))
        data = data[np.arange(0, n_samples, 3),channel].astype(np.int32)
        self.data = data/data.max()
        peaks,_ = scipy.signal.find_peaks(np.diff(self.data), height=self.threshold)
        self.timestep = np.arange(0, n_samples, 3)/fs
        peaks+=1
        self.ttl = pd.Series(index = self.timestep[peaks], data = self.data[peaks])    

    def plot_ttl(self):
        self.graphWidget.clear()        
        self.graphWidget.plot(self.timestep, self.data)
        self.graphWidget.plot(self.ttl.index.values, self.ttl.values, pen=None, symbol='o')
        self.graphWidget.addLine(x=None, y = self.threshold)
        self.graphWidget.setLabel('bottom', 'Time (s)')

    def change_threshold(self, thr):
        self.threshold = thr
        peaks,_ = scipy.signal.find_peaks(np.diff(self.data), height=self.threshold)
        peaks+=1
        self.ttl = pd.Series(index = self.timestep[peaks], data = self.data[peaks])
        self.plot_ttl()

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
            self.epochs = pd.read_csv(path[0], header=None)
            if len(self.epochs.columns) == 3:
                self.epochs.columns = ['start', 'end', 'label']
            elif len(self.epochs.columns) == 2:
                self.epochs.columns = ['start', 'end']
                self.epochs['label'] = ''
            self.setRowCount(0)
            for r in self.epochs.index:
                row = self.rowCount()
                self.insertRow(row)
                for column, clabel in enumerate(self.epochs.columns):
                    item = QTableWidgetItem(str(self.epochs.loc[r,clabel]))
                    self.setItem(row, column, item)
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
        try:
            experimenter=getpass.getuser()

        except:
            experimenter=''

        self.session_information = {
            'path':path,
            'name':'',
            'description':'',
            'experimenter':experimenter,
            'lab':'',
            'institution':'',
        }
        self.subject_information = {
            'age':'',
            'description':'',
            'genotype':'',
            'sex':'',
            'species':'',
            'subject_id':'',
            'weight':'',
            # 'date_of_birth':'',
            'strain':''
        }
                
        self.layout = QHBoxLayout(self)

        formsession = QGroupBox("Session")
        self.form1 = QFormLayout()
        for k in self.session_information.keys():
            if k != 'path':
                tmp = QLineEdit(self.session_information[k])
                self.form1.addRow(k, tmp)
                if k == 'name':
                    self.name = tmp

        formsession.setLayout(self.form1)

        formsubject = QGroupBox("Subject")
        self.form2 = QFormLayout()
        for k in self.subject_information.keys():
            self.form2.addRow(QLabel(k), QLineEdit(self.subject_information[k]))
        formsubject.setLayout(self.form2)

        self.layout.addWidget(formsession)
        self.layout.addWidget(formsubject)

        self.update_path_info(path)

        # self.retrieve_session_information()

    def retrieve_session_information(self):
        n = self.form1.rowCount()
        for i in range(0, n*2, 2):
            key = self.form1.itemAt(i)
            value = self.form1.itemAt(i+1)
            self.session_information[key.widget().text()] = value.widget().text()

        return self.session_information

    def retrieve_subject_information(self):
        n = self.form2.rowCount()
        for i in range(0, n*2, 2):
            key = self.form2.itemAt(i)
            value = self.form2.itemAt(i+1)
            self.subject_information[key.widget().text()] = value.widget().text()

        return self.subject_information
        
    def update_path_info(self, path):
        self.session_information['path'] = path 
        self.session_information['name'] = os.path.basename(path) if path else ''
        self.name.setText(self.session_information['name'])


class TrackingTab(QWidget):
    def __init__(self, path=None, parent=None):
        super(TrackingTab, self).__init__(parent)
        self.path = path
        self.time_units = 's'
        self.tracking_method = 'Optitrack'
        self.track_frequency = 120.0
        self.alignement_csv = 'global' # local or ttl
        self.csv_files = []

        self.align_to_headers = {
            'global':['CSV files'],
            'local':['CSV files','Start\nepoch'],
            'ttl':['CSV files', 'TTL file', 'Number\nof\nchannels','Tracking\nchannel','Bytes\nsize','TTL\nsampling\nfrequency\n(Hz)','Start\nepoch','TTL\ndetection']
                }

        self.headers_to_param = {
            'global':{
                'CSV files':'csv'
            },
            'local':{
                'CSV files':'csv',
                'Start\nepoch':'epoch'
            },
            'ttl':{'CSV files':'csv',
                'TTL file':'ttl',
                'Number\nof\nchannels':'n_channels',
                'Tracking\nchannel':'tracking_channel',
                'Bytes\nsize':'bytes_size',
                'TTL\nsampling\nfrequency\n(Hz)':'fs',
                'Start\nepoch':'epoch',
                'TTL\ndetection':'threshold'
                }
            }

        self.parameters = pd.DataFrame(columns = ['csv'])

        self.ttl_param_widgets = {}

        self.layout = QVBoxLayout(self)
        
        laytop = QHBoxLayout()
        laytop.addWidget(QLabel("Tracking system: "))
        combobox1 = QComboBox()
        combobox1.addItems(['Optitrack', 'Deep Lab Cut', 'Default'])
        combobox1.currentTextChanged.connect(self.get_tracking_method)
        laytop.addWidget(combobox1)

        # Select type of alignement
        abox = QHBoxLayout()
        abox.addWidget(QLabel("Tracking alignment :"))
        abox.setAlignment(Qt.AlignLeft)
        vbox2 = QVBoxLayout()
        for tu in ['Global timestamps in CSV', 'Local timestamps in CSV', 'TTL detection']:
            tubox = QRadioButton(tu, self)
            if tu == 'Global timestamps in CSV': tubox.setChecked(True)
            tubox.toggled.connect(self._update_table_headers)
            vbox2.addWidget(tubox)
        abox.addLayout(vbox2)
        laytop.addLayout(abox)
        
        laytop.addWidget(QLabel("Tracking\nfrequency (Hz): "))
        fs = QDoubleSpinBox()
        fs.setMaximum(100000.0)
        fs.setSingleStep(10.0)
        fs.setValue(self.track_frequency)
        fs.valueChanged.connect(self.get_tracking_frequency)
        laytop.addWidget(fs)

        # Load a csv
        button = QPushButton("Load csv file(s)")
        button.clicked.connect(self.load_csv_files)
        button.resize(button.sizeHint())
        laytop.addWidget(button)

        laytop.addStretch()

        self.layout.addLayout(laytop)

        # Table view
        self.table = QTableWidget(1,1)
        self.table.setHorizontalHeaderLabels(['CSV file'])            
        self.table.itemDoubleClicked.connect(self.change_file)
        self.table.itemChanged.connect(self.change_ttl_params)

        header = self.table.horizontalHeader()       
        header.setSectionResizeMode(QHeaderView.Stretch)

        self.layout.addWidget(self.table)

        self.layout.addStretch()

    def _update_table_headers(self, s):
        rbtn = self.sender()
        if rbtn.isChecked() == True:

            self.table.clearContents()      
            self.parameters = pd.DataFrame(columns = ['csv'])

            nc = self.table.columnCount()

            while nc > 1:
                self.table.removeColumn(nc-1)
                nc -= 1

            if rbtn.text() == 'Global timestamps in CSV':                 
                self.alignement_csv='global'
            elif rbtn.text() == 'Local timestamps in CSV':
                self.alignement_csv='local'
                for i in range(len(self.align_to_headers['local'][1:])):
                    self.table.insertColumn(i+1)
                self.table.setHorizontalHeaderLabels(self.align_to_headers[self.alignement_csv])
            elif rbtn.text() == 'TTL detection':                 
                self.alignement_csv='ttl'
                for i in range(len(self.align_to_headers['ttl'][1:])):
                    self.table.insertColumn(i+1)
                self.table.setHorizontalHeaderLabels(self.align_to_headers[self.alignement_csv])

    def load_csv_files(self, s):
        suggested_dir = self.path if self.path else os.getenv('HOME')
        paths = QFileDialog.getOpenFileNames(self, 'Open CSV', suggested_dir, 'CSV(*.csv)')        
        self.csv_files = paths[0]
        self.table.setRowCount(len(self.csv_files))

        if self.alignement_csv == 'global':

            for i in range(len(self.csv_files)):
                path = os.path.dirname(self.csv_files[i])
                files = os.listdir(os.path.dirname(self.csv_files[i]))
                filename = os.path.basename(self.csv_files[i])

                # Updating parameters table
                self.parameters.loc[filename, 'csv'] = self.csv_files[i]            

                # Set filename in place
                self.table.setItem(i, 0, QTableWidgetItem(filename))

        if self.alignement_csv == 'local':

            self.parameters = pd.DataFrame(columns = ['csv', 'epoch'])

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
                self.table.setCellWidget(i, 1, nepoch)


        elif self.alignement_csv == 'ttl':

            self.parameters = pd.DataFrame(columns = ['csv','ttl','n_channels','tracking_channel','bytes_size','fs','epoch','threshold'])

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
                # nepoch = QSpinBox()
                if n.isdigit(): 
                    # nepoch.setValue(int(n))
                    self.parameters.loc[filename,'epoch'] = int(n)
                    item = QTableWidgetItem(str(n))
                    self.table.setItem(i, 6, item)
                # nepoch.valueChanged.connect(self.change_ttl_params)
                # self.table.setCellWidget(i, 6, nepoch)

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
        w = TTLDetection(
            parent = self,
            file=self.parameters.loc[self.parameters.index[r],'ttl'],
            n_channels=self.parameters.loc[self.parameters.index[r],'n_channels'],
            channel=self.parameters.loc[self.parameters.index[r],'tracking_channel'],
            bytes_size=self.parameters.loc[self.parameters.index[r],'bytes_size'],
            fs=self.parameters.loc[self.parameters.index[r],'fs']
            )
        w.show()
        if w.exec():
            self.parameters['threshold'][r] = w.threshold        
        w.close()
        print(self.parameters['threshold'][r])

    def fill_default_value_parameters(self, filename, row):        
        # ttl_param_widgets = {}
        iterates = zip(
            ['n_channels', 'tracking_channel', 'bytes_size', 'fs'],
            [2,3,4,5],
            [1, 0, 2, 20000.0]
            )        
        for key, col, dval in iterates:
            if col in [2,3,4]:
                self.parameters.loc[filename,key] = int(dval)
            else:
                self.parameters.loc[filename,key] = dval
            item = QTableWidgetItem(str(dval))
            self.table.setItem(row, col, item)
            # if key == 'fs':
            #     ttl_param_widgets[key] = QDoubleSpinBox()
            #     ttl_param_widgets[key].setMaximum(100000.0)
            #     ttl_param_widgets[key].setSingleStep(1000.0)
            # else:
            #     ttl_param_widgets[key] = QSpinBox()
            # ttl_param_widgets[key].setValue(dval)
            # ttl_param_widgets[key].valueChanged.connect(self.change_ttl_params)
            # self.table.setCellWidget(row, col, ttl_param_widgets[key])
            # Adding default trheshod value
            self.parameters.loc[filename,'threshold'] = 0.3
        # self.ttl_param_widgets[row] = ttl_param_widgets

    def change_file(self, item):
        if self.table.currentColumn() in [0, 1]:
            suggested_dir = self.path if self.path else os.getenv('HOME')
            if self.table.currentColumn() == 0:
                paths = QFileDialog.getOpenFileName(self, 'Open CSV', suggested_dir, 'CSV(*.csv)')
            if self.table.columnCount()>1:        
                if self.table.currentColumn() == 1:
                    paths = QFileDialog.getOpenFileName(self, 'Open TTL file', suggested_dir)
            filename = os.path.basename(paths[0])
            item.setText(filename)
            self.parameters.iloc[self.table.currentRow(),self.table.currentColumn()] = paths[0]

    def get_tracking_method(self, s):
        self.tracking_method = s

    def get_tracking_frequency(self, s):
        self.track_frequency = float(s)        

    def change_ttl_params(self, item):
        row, col = (self.table.currentRow(),self.table.currentColumn())
        if col in [2, 3, 4, 5, 6]:
            if col == 5: 
                self.parameters.iloc[row, col] = float(item.text())
            else:
                self.parameters.iloc[row, col] = int(item.text())        

    def update_path_info(self, path):
        self.path = path


class BaseLoaderGUI(QMainWindow):

    def __init__(self, path=None):
        super().__init__()

        # Basic properties to return
        self.status = False
        self.path = path

        self.setWindowTitle("The minimalist session loader")
        self.setMinimumSize(900, 560)

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
        # self.tab_help = HelpTab()
        
        self.tabs.addTab(self.tab_session, 'Session Information')
        self.tabs.addTab(self.tab_epoch, 'Epochs')
        self.tabs.addTab(self.tab_tracking, 'Tracking')
        # self.tabs.addTab(self.tab_help, 'Help')

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
        self.session_information = self.tab_session.retrieve_session_information()
        self.subject_information = self.tab_session.retrieve_subject_information()
        self.epochs = self.tab_epoch.table.epochs
        self.time_units_epochs = self.tab_epoch.time_units
        self.tracking_parameters = self.tab_tracking.parameters
        self.tracking_alignement = self.tab_tracking.alignement_csv
        self.tracking_method = self.tab_tracking.tracking_method
        self.tracking_frequency = self.tab_tracking.track_frequency
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





