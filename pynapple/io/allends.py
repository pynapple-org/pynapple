"""
Class and functions for loading data from Allen Brain Atlas via the Allen Software Development Kit (allenSDK) API
Currently only supports "Visual Coding - Neuropixels" database.

The Visual Coding - Neuropixels project uses high-density extracellular electrophysiology probes to record spikes from
a wide variety of regions in the mouse brain. Experiments were designed to study the activity of the visual cortex
and thalamus in the context of passive visual stimuluation, but these data can be used to address a wide variety of topics.

To see more, see: https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html

@author: Selen Calgin
Date: 06/19/2023
Last edit: 07/06/2023 by Selen Calgin

"""
import json
import os
import tkinter as tk
from tkinter import ttk

from allensdk.brain_observatory.ecephys.ecephys_project_cache import (
    EcephysProjectCache,
)

from .. import core as nap
from .loader import BaseLoader


class AllenDS(BaseLoader):
    """
    Loader for Allen Brain Atlas data. Currenlty only supports Neuropixels data.
    AllenDS = "Allen Data Set"
    """

    def __init__(self, path):
        """

        Parameters
        ----------
        path (str):
            path to where nwbfile and data will be saved

        """

        # path where data cache will be stored
        self.cache_path = os.path.join(
            path, "pynallensdk"
        )  # directory where Allens data is downloaded

        # make manifest file specific for pynapple
        manifest_path = os.path.join(self.cache_path, "manifest.json")
        self._write_manifest_file(manifest_path)

        # init data cache
        self.cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

        # get session ID from user
        self.session_id = self._get_session_id(self.cache)
        self.session_path = os.path.join(
            self.cache_path, "session_%d" % self.session_id
        )

        # download session data
        # note: if data has been previously downloaded, data won't be redownloaded (allensdk API is smart)
        try:
            self.session = self.cache.get_session_data(self.session_id)

        except ValueError:
            raise ValueError("Unexpected: session ID %d not found." % self.session_id)

        # initialize nwb variables
        self.nwb_path = os.path.join(self.session_path, "pynapplenwb")
        self.nwbfilename = "session_%d.nwb" % self.session_id
        self.nwbfilepath = os.path.join(self.nwb_path, self.nwbfilename)

        # load data
        self.load_epochs()
        self.load_stimulus_epochs()
        self.load_optogenetic_stimulus_epochs()
        self.load_spikes()
        self.load_metadata()

    def load_epochs(self):
        """
        Load session epochs into Interval set.
        Epoch in this context is simply the entire recording session, as reference.
        Currently, uses last time for the last stimulus, however this could be refined.

        """
        start = 0
        stop = self.session.optogenetic_stimulation_epochs.iloc[-1]["stop_time"]
        self.epochs = {
            "session": nap.IntervalSet(start=start, end=stop, time_units="s")
        }
        # global time support of data
        self.time_support = nap.IntervalSet(start=start, end=stop, time_units="s")

    def load_stimulus_epochs(self):
        """
        Loads stimulus epochs labeled by stimulus name and by block
        from allen database to pynapple workspace
        by converting dataframe to Interval Set

        """
        stimulus_epochs = self.session.get_stimulus_epochs()

        # rename columns
        # label is stimulus name
        stimulus_epochs = stimulus_epochs.rename(
            columns={
                "start_time": "start",
                "stop_time": "end",
                "stimulus_name": "label",
            }
        )
        self.stimulus_epochs_names = self._make_epochs(stimulus_epochs)

        # rename columns
        # label is stimulus block
        stimulus_epochs = stimulus_epochs.drop(labels="label", axis=1)
        stimulus_epochs = stimulus_epochs.rename(columns={"stimulus_block": "label"})
        self.stimulus_epochs_block = self._make_epochs(stimulus_epochs)

    def load_optogenetic_stimulus_epochs(self):
        """
        Load optogenetic stimulus epochs into IntervalSet.

        """
        optogenetic_epochs = self.session.optogenetic_stimulation_epochs
        optogenetic_epochs = optogenetic_epochs.rename(
            columns={
                "start_time": "start",
                "stop_time": "end",
                "stimulus_name": "label",
            }
        )
        self.optogenetic_stimulus_epochs = self._make_epochs(optogenetic_epochs)

    def load_spikes(self):
        """
        Extract spike times and load to pynapple workspace as TsGroup
        """
        spike_times = self.session.spike_times
        spikes = {n: nap.Ts(t=spike_times[n], time_units="s") for n in spike_times}

        self.spikes = nap.TsGroup(
            spikes,
            time_support=self.time_support,
            time_units="s",
            **self.session.units.sort_index(),
        )

    def load_metadata(self):
        """
        Loading metadata for stimulus conditions/presentations and channel/probe information
        I.e. any useful information that isn't compatible with pynapple objects
        Could use refining or additional metadata loading

        Currently loading:
        stimulus_presentations = all stimulus presentations & metadata
        stimulus_conditions = unique stimulus conditions, more info on stimulus
        probes
        channels
        """
        self.metadata = {
            "stimulus_presentations": self.session.stimulus_presentations,
            "stimulus_conditions": self.session.stimulus_conditions,
            "probes": self.session.probes,
            "channels": self.session.channels,
        }

    def _write_manifest_file(self, path):
        """
        Writes manifest.json file that is specifically compatible for pynapple use.
        In particular, changes one line from default manifest.json from Allen,
        such that nwb file is saved under "pynapplenwb" folder.

        Parameters
        ----------
        path to where manifest file is saved

        """
        manifest_data = {
            "manifest": [
                {"type": "manifest_version", "value": "0.3.0"},
                {"key": "BASEDIR", "type": "dir", "spec": "."},
                {
                    "key": "sessions",
                    "type": "file",
                    "spec": "sessions.csv",
                    "parent_key": "BASEDIR",
                },
                {
                    "key": "probes",
                    "type": "file",
                    "spec": "probes.csv",
                    "parent_key": "BASEDIR",
                },
                {
                    "key": "channels",
                    "type": "file",
                    "spec": "channels.csv",
                    "parent_key": "BASEDIR",
                },
                {
                    "key": "units",
                    "type": "file",
                    "spec": "units.csv",
                    "parent_key": "BASEDIR",
                },
                {
                    "key": "session_data",
                    "type": "dir",
                    "spec": "session_%d/pynapplenwb",
                    "parent_key": "BASEDIR",
                },
                {
                    "key": "session_nwb",
                    "type": "file",
                    "spec": "session_%d.nwb",
                    "parent_key": "session_data",
                },
                {
                    "key": "session_analysis_metrics",
                    "type": "file",
                    "spec": "session_%d_analysis_metrics.csv",
                    "parent_key": "session_data",
                },
                {
                    "key": "probe_lfp_nwb",
                    "type": "file",
                    "spec": "probe_%d_lfp.nwb",
                    "parent_key": "session_data",
                },
                {
                    "key": "movie_dir",
                    "type": "dir",
                    "spec": "natural_movie_templates",
                    "parent_key": "BASEDIR",
                },
                {
                    "key": "typewise_analysis_metrics",
                    "type": "file",
                    "spec": "%s_analysis_metrics.csv",
                    "parent_key": "BASEDIR",
                },
                {
                    "key": "natural_movie",
                    "type": "file",
                    "spec": "natural_movie_%d.h5",
                    "parent_key": "movie_dir",
                },
                {
                    "key": "natural_scene_dir",
                    "type": "dir",
                    "spec": "natural_scene_templates",
                    "parent_key": "BASEDIR",
                },
                {
                    "key": "natural_scene",
                    "type": "file",
                    "spec": "natural_scene_%d.tiff",
                    "parent_key": "natural_scene_dir",
                },
            ]
        }

        manifest_json = json.dumps(manifest_data, indent=4)

        if not os.path.exists(path):
            with open(path, "w") as file:
                file.write(manifest_json)

    @staticmethod
    def _get_session_id(cache):
        """
        Create dropdown menu of sessions for user to select
        Future dev: add more dataset options; currently only supports Neuropixels
        Parameters
        ----------
        cache: EcephysProjectCache
            Allensdk API for Neuropixels

        Returns: int
            session id through user selection
        -------

        """
        # Define the session IDs for type 1 and type 2 sessions
        sessions = cache.get_session_table()
        type1_sessions = sessions[
            sessions["session_type"] == "brain_observatory_1.1"
        ].index.to_list()
        type2_sessions = sessions[
            sessions["session_type"] == "functional_connectivity"
        ].index.to_list()

        def type_selected(event):
            selected_type = type_var.get()

            # Update the options of the session dropdown menu based on the selected type
            if selected_type == "Brain observatory":
                session_dropdown["values"] = type1_sessions
            elif selected_type == "Functional connectivity":
                session_dropdown["values"] = type2_sessions

        def ok_button_click():
            global session_id
            session_id = session_var.get()

            # Validate the selection
            if session_id and session_id.isdigit():
                # Close the GUI
                window.destroy()

            else:
                # Display an error message if no selection is made
                error_label.config(text="Please select a session ID.", foreground="red")

        # Create the Tkinter window
        window = tk.Tk()
        window.title("Session Selection")

        # Make the window appear on the screen
        window.attributes("-topmost", True)

        # Create a label and dropdown menu for selecting the session type
        type_label = ttk.Label(window, text="Select Session Type:")
        type_label.pack(pady=10)

        type_var = tk.StringVar()
        type_var.set("Type 1")

        type_dropdown = ttk.OptionMenu(
            window,
            type_var,
            "Brain observatory",
            "Brain observatory",
            "Functional " "connectivity",
            command=type_selected,
        )
        type_dropdown.pack(pady=5)

        # Create a label and dropdown menu for selecting the session ID
        session_label = ttk.Label(window, text="Select Session ID:")
        session_label.pack(pady=10)

        session_var = tk.StringVar()

        session_dropdown = ttk.Combobox(
            window, textvariable=session_var, state="readonly"
        )
        session_dropdown.pack(pady=5)

        # Initially, populate the session dropdown with type 1 sessions
        session_dropdown["values"] = type1_sessions

        # Create an OK button to capture the user selection and close the GUI
        ok_button = ttk.Button(window, text="OK", command=ok_button_click)
        ok_button.pack(pady=10)

        # Create a label for displaying error messages
        error_label = ttk.Label(window, text="", foreground="red")
        error_label.pack(pady=5)

        window.mainloop()

        return int(session_id)
