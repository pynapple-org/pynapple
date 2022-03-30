# -*- coding: utf-8 -*-
"""
Loaders for calcium imaging data with miniscope.
Support CNMF-E in matlab, inscopix-cnmfe and minian.

"""
# @Author: gviejo
# @Date:   2022-02-17 11:07:00
# @Last Modified by:   gviejo
# @Last Modified time: 2022-03-27 22:35:52

import os
from .loader import BaseLoader
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ophys import TwoPhotonSeries, OpticalChannel, ImageSegmentation, Fluorescence, CorrectedImageStack, MotionCorrection, RoiResponseSeries
import pandas as pd
import numpy as np
from .. import core as nap
from PyQt5.QtWidgets import QApplication
from .ophys_gui import OphysGUI
import tifffile as tiff
import zarr
from scipy.io.matlab import loadmat

class CNMF_E(BaseLoader):
    """Loader for data processed with matlab CNMF-E(https://github.com/zhoupc/CNMF_E).
    The path folder should contain a file ending in .mat
    when calling Source2d.save_neurons
    
    Attributes
    ----------
    A : numpy.ndarray
        Spatial footprints
    C : TsdFrame
        The calcium transients
    sampling_rate : float
        Sampling rate of the data (default is 30 Hz).
    
    """
    
    def __init__(self, path):
        """        
        
        Parameters
        ----------
        path : str
            The path to the data.
        """     
        self.basename = os.path.basename(path)        
        
        super().__init__(path)

        # Need to check if nwb file exists and if data are there
        loading_my_data = True
        if self.path is not None:
            nwb_path = os.path.join(self.path, 'pynapplenwb')
            if os.path.exists(nwb_path):
                files = os.listdir(nwb_path)
                if len([f for f in files if f.endswith('.nwb')]):                    
                    success = self.load_cnmfe_nwb(path)
                    if success: loading_my_data = False

        # Bypass if data have already been transfered to nwb
        if loading_my_data:

            app = QApplication([])
            window = OphysGUI(path=path)
            window.show()
            app.exec()
            if window.status:
                self.ophys_information = window.ophys_information
                self.load_cnmf_e(path)
                self.save_cnmfe_nwb(path)

    def load_cnmf_e(self, path):
        """
        Load the calcium transients and the spatial footprints.

        Parameters
        ----------
        path : str
            Path to the session
        """        
        files = os.listdir(path)
        matfiles = [f for f in files if f.endswith('.mat')]

        if len(matfiles):
            data = loadmat(os.path.join(path, matfiles[0]), struct_as_record=False)
        else:
            raise RuntimeError("No mat file found in {}".format(path))

        self.struct = data['neuron_results'][0][0]

        C = self.struct.C.T
        self.A = self.struct.A.T

        self.sampling_rate = float(self.ophys_information['ImagingPlane']['imaging_rate'])

        time_index = np.arange(0, len(C))/self.sampling_rate

        self.C = nap.TsdFrame(
            t = time_index,
            d = C
            )

        return None

    def save_cnmfe_nwb(self, path):
        """
        Save the data to NWB.
        Since there is no one-photon field in nwb, it uses the two-photon field.
        
        Parameters
        ----------
        path : TYPE
            Description
        """
        self.nwb_path = os.path.join(path, 'pynapplenwb')
        if os.path.exists(self.nwb_path):
            files = os.listdir(self.nwb_path)
        else:
            raise RuntimeError("Path {} does not exist.".format(self.nwb_path))
        self.nwbfilename = [f for f in os.listdir(self.nwb_path) if 'nwb' in f][0]
        self.nwbfilepath = os.path.join(self.nwb_path, self.nwbfilename)

        io = NWBHDF5IO(self.nwbfilepath, 'r+')
        nwbfile = io.read()
        
        device_info = self.ophys_information['device']
        device = nwbfile.create_device(
            name=device_info['name'],
            description=device_info['description'],
            manufacturer=device_info['manufacturer']
        )
        optical_info = self.ophys_information['OpticalChannel']
        optical_info['emission_lambda'] = float(optical_info['emission_lambda'])
        optical_channel = OpticalChannel(
            name=optical_info['name'],
            description=optical_info['description'],
            emission_lambda=optical_info['emission_lambda']
        )
        imaging_info = self.ophys_information['ImagingPlane']
        imaging_info['excitation_lambda'] = float(imaging_info['excitation_lambda'])
        imaging_plane = nwbfile.create_imaging_plane(
            name=imaging_info['name'],
            optical_channel=optical_channel,
            imaging_rate=self.sampling_rate,
            description=imaging_info['description'],
            device=device,
            excitation_lambda=imaging_info['excitation_lambda'],
            indicator=imaging_info['indicator'],
            location=imaging_info['location'],
        )

        ophys_module = nwbfile.create_processing_module(
            name='ophys',
            description='optical physiology processed data'
        )

        seg_info = self.ophys_information['PlaneSegmentation']
        img_seg = ImageSegmentation()
        ps = img_seg.create_plane_segmentation(
            name=seg_info['name'],
            description=seg_info['description'],
            imaging_plane=imaging_plane,            
        )

        for i in range(self.C.shape[1]):
            image_mask = np.atleast_2d(self.A[i])
            # add image mask to plane segmentation
            ps.add_roi(image_mask=image_mask)


        ophys_module.add(img_seg)

        rt_region = ps.create_roi_table_region(
            region=list(np.arange(self.C.shape[1])),
            description='ROIs'
        )

        roi_resp_series = RoiResponseSeries(
            name='RoiResponseSeries',
            data=self.C.values,
            rois=rt_region,
            unit='lumens',            
            timestamps = self.C.index.values
        )

        fl = Fluorescence(roi_response_series=roi_resp_series)
        ophys_module.add(fl)
        
        io.write(nwbfile)
        io.close()

        return
        
    def load_cnmfe_nwb(self, path):
        """
        Load the calcium transient and spatial footprint from nwb
        
        Parameters
        ----------
        path : str
            Path to the session
        """
        self.nwb_path = os.path.join(path, 'pynapplenwb')
        if os.path.exists(self.nwb_path):
            files = os.listdir(self.nwb_path)
        else:
            raise RuntimeError("Path {} does not exist.".format(self.nwb_path))
        self.nwbfilename = [f for f in os.listdir(self.nwb_path) if 'nwb' in f][0]
        self.nwbfilepath = os.path.join(self.nwb_path, self.nwbfilename)

        io = NWBHDF5IO(self.nwbfilepath, 'r')
        nwbfile = io.read()

        if 'ophys' in nwbfile.processing.keys():
            data = nwbfile.processing['ophys']['Fluorescence']['RoiResponseSeries'].data[:]
            t = nwbfile.processing['ophys']['Fluorescence']['RoiResponseSeries'].timestamps[:]
            self.C = nap.TsdFrame(
                t = t,
                d = data
                )
            self.A = nwbfile.processing['ophys']['ImageSegmentation']['PlaneSegmentation']['image_mask'].data[:]

            io.close()
            return True
        else:            
            io.close()
            return False

class Minian(BaseLoader):
    """Loader for data processed with Minian (https://github.com/denisecailab/minian).
    The path folder should contain a subfolder name minian.    
    
    Attributes
    ----------
    A : numpy.ndarray
        Spatial footprints
    C : TsdFrame
        The calcium transients
    sampling_rate : float
        Sampling rate of the data (default is 30 Hz).
    
    """
    
    def __init__(self, path):
        """        
        
        Parameters
        ----------
        path : str
            The path to the data.
        """     
        self.basename = os.path.basename(path)        
        
        super().__init__(path)

        # Need to check if nwb file exists and if data are there
        loading_my_data = True
        if self.path is not None:
            nwb_path = os.path.join(self.path, 'pynapplenwb')
            if os.path.exists(nwb_path):
                files = os.listdir(nwb_path)
                if len([f for f in files if f.endswith('.nwb')]):                    
                    success = self.load_cnmfe_nwb(path)
                    if success: loading_my_data = False

        # Bypass if data have already been transfered to nwb
        if loading_my_data:

            app = QApplication([])
            window = OphysGUI(path=path)
            window.show()
            app.exec()
            if window.status:
                self.ophys_information = window.ophys_information
                self.load_minian(path)
                self.save_cnmfe_nwb(path)

    def load_minian(self, path):
        """
        Load the calcium transients and the spatial footprints.

        Parameters
        ----------
        path : str
            Path to the session
        """        
        minian_folder = os.path.join(path, 'minian')

        if not os.path.exists(minian_folder):
            raise RuntimeError("Path {} does not contain a minian folder".format(path))

        data = zarr.open(minian_folder, 'r')

        C = data['C.zarr']['C'][:]
        C = C.T
        self.sampling_rate = float(self.ophys_information['ImagingPlane']['imaging_rate'])
        time_index = np.arange(0, len(C))/self.sampling_rate

        self.C = nap.TsdFrame(
            t = time_index,
            d = C
            )

        self.A = data['A.zarr']['A'][:]
        
        return None

    def save_cnmfe_nwb(self, path):
        """
        Save the data to NWB.
        Since there is no one-photon field in nwb, it uses the two-photon field.
        
        Parameters
        ----------
        path : TYPE
            Description
        """
        self.nwb_path = os.path.join(path, 'pynapplenwb')
        if os.path.exists(self.nwb_path):
            files = os.listdir(self.nwb_path)
        else:
            raise RuntimeError("Path {} does not exist.".format(self.nwb_path))
        self.nwbfilename = [f for f in os.listdir(self.nwb_path) if 'nwb' in f][0]
        self.nwbfilepath = os.path.join(self.nwb_path, self.nwbfilename)

        io = NWBHDF5IO(self.nwbfilepath, 'r+')
        nwbfile = io.read()
        
        device_info = self.ophys_information['device']
        device = nwbfile.create_device(
            name=device_info['name'],
            description=device_info['description'],
            manufacturer=device_info['manufacturer']
        )
        optical_info = self.ophys_information['OpticalChannel']
        optical_info['emission_lambda'] = float(optical_info['emission_lambda'])
        optical_channel = OpticalChannel(
            name=optical_info['name'],
            description=optical_info['description'],
            emission_lambda=optical_info['emission_lambda']
        )
        imaging_info = self.ophys_information['ImagingPlane']
        imaging_info['excitation_lambda'] = float(imaging_info['excitation_lambda'])
        imaging_plane = nwbfile.create_imaging_plane(
            name=imaging_info['name'],
            optical_channel=optical_channel,
            imaging_rate=self.sampling_rate,
            description=imaging_info['description'],
            device=device,
            excitation_lambda=imaging_info['excitation_lambda'],
            indicator=imaging_info['indicator'],
            location=imaging_info['location'],
        )

        ophys_module = nwbfile.create_processing_module(
            name='ophys',
            description='optical physiology processed data'
        )

        seg_info = self.ophys_information['PlaneSegmentation']
        img_seg = ImageSegmentation()
        ps = img_seg.create_plane_segmentation(
            name=seg_info['name'],
            description=seg_info['description'],
            imaging_plane=imaging_plane,            
        )

        for i in range(self.C.shape[1]):
            image_mask = self.A[i]
            # add image mask to plane segmentation
            ps.add_roi(image_mask=image_mask)


        ophys_module.add(img_seg)

        rt_region = ps.create_roi_table_region(
            region=list(np.arange(self.C.shape[1])),
            description='ROIs'
        )

        roi_resp_series = RoiResponseSeries(
            name='RoiResponseSeries',
            data=self.C.values,
            rois=rt_region,
            unit='lumens',            
            timestamps = self.C.index.values
        )

        fl = Fluorescence(roi_response_series=roi_resp_series)
        ophys_module.add(fl)
        
        io.write(nwbfile)
        io.close()

        return
        
    def load_cnmfe_nwb(self, path):
        """
        Load the calcium transient and spatial footprint from nwb
        
        Parameters
        ----------
        path : str
            Path to the session
        """
        self.nwb_path = os.path.join(path, 'pynapplenwb')
        if os.path.exists(self.nwb_path):
            files = os.listdir(self.nwb_path)
        else:
            raise RuntimeError("Path {} does not exist.".format(self.nwb_path))
        self.nwbfilename = [f for f in os.listdir(self.nwb_path) if 'nwb' in f][0]
        self.nwbfilepath = os.path.join(self.nwb_path, self.nwbfilename)

        io = NWBHDF5IO(self.nwbfilepath, 'r')
        nwbfile = io.read()

        if 'ophys' in nwbfile.processing.keys():
            data = nwbfile.processing['ophys']['Fluorescence']['RoiResponseSeries'].data[:]
            t = nwbfile.processing['ophys']['Fluorescence']['RoiResponseSeries'].timestamps[:]
            self.C = nap.TsdFrame(
                t = t,
                d = data
                )
            self.A = nwbfile.processing['ophys']['ImageSegmentation']['PlaneSegmentation']['image_mask'].data[:]

            io.close()
            return True
        else:            
            io.close()
            return False

class InscopixCNMFE(BaseLoader):
    """Loader for Inscopix-cnmfe (https://github.com/inscopix/inscopix-cnmfe).
    The folder should contain a file ending with '_traces.csv'
    and a tiff file for spatial footprints.
    
    Attributes
    ----------
    A : np.ndarray
        The spatial footprints
    C : TsdFrame
        The calcium transients
    sampling_rate : float
        Sampling rate of the data (default is 30 Hz).
    
    """
    
    def __init__(self, path):
        """        
        
        Parameters
        ----------
        path : str
            The path to the data.
        """     
        self.basename = os.path.basename(path)        
        
        super().__init__(path)

        # Need to check if nwb file exists and if data are there
        loading_my_data = True
        if self.path is not None:
            nwb_path = os.path.join(self.path, 'pynapplenwb')
            if os.path.exists(nwb_path):
                files = os.listdir(nwb_path)
                if len([f for f in files if f.endswith('.nwb')]):                    
                    success = self.load_cnmfe_nwb(path)
                    if success: loading_my_data = False

        # Bypass if data have already been transfered to nwb
        if loading_my_data:

            app = QApplication([])
            window = OphysGUI(path=path)
            window.show()
            app.exec()
            if window.status:
                self.ophys_information = window.ophys_information
                self.load_inscopix_cnmfe(path)
                self.save_cnmfe_nwb(path)

    def load_inscopix_cnmfe(self, path):
        """
        Load the calcium transients and the spatial footprints.
        
        Parameters
        ----------
        path : str
            Path to the session
        """        
        files = os.listdir(path)
        tracefile = [f for f in files if f.endswith('_traces.csv')]

        if len(tracefile):
            C = pd.read_csv(os.path.join(path, tracefile[0]), index_col = 0)
        else:
            raise RuntimeError("Path {} does not contain the file {}".format(path, '*_traces.csv'))

        self.sampling_rate = float(self.ophys_information['ImagingPlane']['imaging_rate'])

        time_index = np.arange(0, len(C))/self.sampling_rate

        self.C = nap.TsdFrame(
            t = time_index,
            d = C.values            
            )

        tifffile = [f for f in files if f.endswith('.tiff')]
        if len(tifffile):
            self.A = tiff.imread(os.path.join(path, tifffile[0]))
        else:
            raise RuntimeError("Path {} does not contain the file {}".format(path, "*.tiff"))

        return None

    def save_cnmfe_nwb(self, path):
        """
        Save the data to NWB.
        Since there is no one-photon field in nwb, it uses the two-photon field.
        
        Parameters
        ----------
        path : TYPE
            Description
        """
        self.nwb_path = os.path.join(path, 'pynapplenwb')
        if os.path.exists(self.nwb_path):
            files = os.listdir(self.nwb_path)
        else:
            raise RuntimeError("Path {} does not exist.".format(self.nwb_path))
        self.nwbfilename = [f for f in os.listdir(self.nwb_path) if 'nwb' in f][0]
        self.nwbfilepath = os.path.join(self.nwb_path, self.nwbfilename)

        io = NWBHDF5IO(self.nwbfilepath, 'r+')
        nwbfile = io.read()
        
        device_info = self.ophys_information['device']
        device = nwbfile.create_device(
            name=device_info['name'],
            description=device_info['description'],
            manufacturer=device_info['manufacturer']
        )
        optical_info = self.ophys_information['OpticalChannel']
        optical_info['emission_lambda'] = float(optical_info['emission_lambda'])
        optical_channel = OpticalChannel(
            name=optical_info['name'],
            description=optical_info['description'],
            emission_lambda=optical_info['emission_lambda']
        )
        imaging_info = self.ophys_information['ImagingPlane']
        imaging_info['excitation_lambda'] = float(imaging_info['excitation_lambda'])
        imaging_plane = nwbfile.create_imaging_plane(
            name=imaging_info['name'],
            optical_channel=optical_channel,
            imaging_rate=self.sampling_rate,
            description=imaging_info['description'],
            device=device,
            excitation_lambda=imaging_info['excitation_lambda'],
            indicator=imaging_info['indicator'],
            location=imaging_info['location'],
        )

        ophys_module = nwbfile.create_processing_module(
            name='ophys',
            description='optical physiology processed data'
        )

        seg_info = self.ophys_information['PlaneSegmentation']
        img_seg = ImageSegmentation()
        ps = img_seg.create_plane_segmentation(
            name=seg_info['name'],
            description=seg_info['description'],
            imaging_plane=imaging_plane,            
        )

        for i in range(self.C.shape[1]):
            image_mask = self.A[i]
            # add image mask to plane segmentation
            ps.add_roi(image_mask=image_mask)


        ophys_module.add(img_seg)

        rt_region = ps.create_roi_table_region(
            region=list(np.arange(self.C.shape[1])),
            description='ROIs'
        )

        roi_resp_series = RoiResponseSeries(
            name='RoiResponseSeries',
            data=self.C.values,
            rois=rt_region,
            unit='lumens',            
            timestamps = self.C.index.values
        )

        fl = Fluorescence(roi_response_series=roi_resp_series)
        ophys_module.add(fl)
        
        io.write(nwbfile)
        io.close()

        return
        
    def load_cnmfe_nwb(self, path):
        """
        Load the calcium transient and spatial footprint from nwb
        
        Parameters
        ----------
        path : str
            Path to the session
        """
        self.nwb_path = os.path.join(path, 'pynapplenwb')
        if os.path.exists(self.nwb_path):
            files = os.listdir(self.nwb_path)
        else:
            raise RuntimeError("Path {} does not exist.".format(self.nwb_path))
        self.nwbfilename = [f for f in os.listdir(self.nwb_path) if 'nwb' in f][0]
        self.nwbfilepath = os.path.join(self.nwb_path, self.nwbfilename)

        io = NWBHDF5IO(self.nwbfilepath, 'r')
        nwbfile = io.read()

        if 'ophys' in nwbfile.processing.keys():
            data = nwbfile.processing['ophys']['Fluorescence']['RoiResponseSeries'].data[:]
            t = nwbfile.processing['ophys']['Fluorescence']['RoiResponseSeries'].timestamps[:]
            self.C = nap.TsdFrame(
                t = t,
                d = data
                )
            self.A = nwbfile.processing['ophys']['ImageSegmentation']['PlaneSegmentation']['image_mask'].data[:]

            io.close()
            return True
        else:            
            io.close()
            return False