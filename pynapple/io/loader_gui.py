# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2023-03-29 15:05:14
# @Last Modified by:   gviejo
# @Last Modified time: 2023-04-06 20:06:47
import getpass
import os
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import pandas as pd


class EntryPopup(ttk.Entry):
    def __init__(self, parent, iid, column, text, table, **kw):
        ttk.Style().configure("pad.TEntry", padding="1 1 1 1")
        super().__init__(parent, style="pad.TEntry", **kw)
        self.tv = parent
        self.iid = iid
        self.column = column
        self.table = table

        self.insert(0, text)
        self["exportselection"] = False

        self.focus_force()
        self.select_all()
        self.bind("<Return>", self.on_return)
        self.bind("<Control-a>", self.select_all)
        self.bind("<Escape>", lambda *ignore: self.destroy())

    def on_return(self, event):
        rowid = self.tv.focus()
        vals = self.tv.item(rowid, "values")
        vals = list(vals)
        vals[self.column] = self.get()
        self.tv.item(rowid, values=vals)
        self.destroy()
        self.table.loc[int(rowid), self.table.columns[int(self.column)]] = vals[
            self.column
        ]

    def select_all(self, *ignore):
        """Set selection on the whole text"""
        self.selection_range(0, "end")

        # returns 'break' to interrupt default key-bindings
        return "break"


class TrackingTable(ttk.Treeview):
    def __init__(self, parent, headers, path, parameters):
        super().__init__(parent, columns=headers[1:])

        self.path = path
        self.parameters = parameters
        self.n_row = 0

        self.name_to_keys = {}

        for i, n in enumerate(headers):
            self.heading(f"#{i}", text=n)
            if i == 0:
                self.column(f"#{i}", anchor=tk.CENTER, stretch=tk.NO, width=0)
            else:
                self.column(f"#{i}", anchor="c", width=10)
                self.name_to_keys[f"#{i}"] = self.parameters.columns[i - 1]

        yscrollbar = ttk.Scrollbar(self, orient="vertical", command=self.yview)
        self.configure(yscrollcommand=yscrollbar.set)
        yscrollbar.pack(side="right", fill="y")

        self.bind("<Double-1>", lambda event: self.on_double_click(event))
        self.tag_configure("oddrow", background="gray66")
        self.tag_configure("evenrow", background="gray92")
        self.tags = ["evenrow", "oddrow"]

    def _update_parameters(self, csv_files, alignment_csv):
        if alignment_csv == "global":
            for i in range(len(csv_files)):
                # path = os.path.dirname(csv_files[i])
                files = os.listdir(os.path.dirname(csv_files[i]))
                filename = os.path.basename(csv_files[i])

                # Updating parameters table
                self.parameters.loc[i, "csv"] = csv_files[i]

                # Set filename in place
                self.insert_row((filename))

        elif alignment_csv == "local":
            for i in range(len(csv_files)):
                # path = os.path.dirname(csv_files[i])
                files = os.listdir(os.path.dirname(csv_files[i]))
                filename = os.path.basename(csv_files[i])

                # Updating parameters table
                self.parameters.loc[i, "csv"] = csv_files[i]

                # Infer the epoch
                n = os.path.splitext(filename)[0].split("_")[-1]
                if n.isdigit():
                    self.parameters.loc[i, "epoch"] = int(n)

                # Set filename in place
                self.insert_row((filename, n))

        elif alignment_csv == "ttl":
            for i in range(len(csv_files)):
                # path = os.path.dirname(csv_files[i])
                files = os.listdir(os.path.dirname(csv_files[i]))
                filename = os.path.basename(csv_files[i])

                # Updating parameters table
                self.parameters.loc[i, "csv"] = csv_files[i]

                # Infer the epoch
                n = os.path.splitext(filename)[0].split("_")[-1]
                epoch = ""
                if n.isdigit():
                    self.parameters.loc[i, "epoch"] = int(n)
                    epoch = str(int(n))

                values = [filename]

                # Infer the ttl file
                possiblettlfile = [f for f in files if "_" + n in f and f != filename]
                if len(possiblettlfile):
                    self.parameters.loc[i, "ttl"] = os.path.join(
                        self.path, possiblettlfile[0]
                    )
                    values.append(possiblettlfile[0])
                else:
                    values.append("Select ttl file")

                # Default analogin parameters
                iterates = zip(
                    [
                        "n_channels",
                        "tracking_channel",
                        "bytes_size",
                        "fs",
                        "epoch",
                        "threshold",
                    ],
                    ["1", "0", "2", "20000.0", epoch, "0.3"],
                )
                for key, dval in iterates:
                    self.parameters.loc[i, key] = dval
                    values.append(dval)

                self.insert_row(values)

    def insert_row(self, values):
        self.insert(
            "", "end", iid=self.n_row, values=values, tags=(self.tags[self.n_row % 2])
        )
        self.n_row += 1

    def on_double_click(self, event):
        # close previous popups
        try:  # in case there was no previous popup
            self.entryPopup.destroy()
        except AttributeError:
            pass

        rowid = self.identify_row(event.y)
        column = self.identify_column(event.x)
        key = self.name_to_keys[column]
        if not rowid:
            return
        if key in ["csv", "ttl"]:
            if key == "csv":
                path = filedialog.askopenfilename(
                    initialdir=self.path,
                    filetypes=[("CSV File", "*.csv")],
                )
            else:
                path = filedialog.askopenfilename(initialdir=self.path)
            self.parameters[key] = path
            vals = self.item(rowid, "values")
            vals = list(vals)
            vals[int(column[1:]) - 1] = os.path.basename(path)
            self.item(rowid, values=vals)
            return
        else:
            x, y, width, height = self.bbox(rowid, column)
            pady = height // 2
            text = self.item(rowid, "values")[int(column[1:]) - 1]
            self.entryPopup = EntryPopup(
                self, rowid, int(column[1:]) - 1, text, self.parameters
            )
            self.entryPopup.place(
                x=x, y=y + pady, width=width, height=height, anchor="w"
            )
            return


class TrackingTab(ttk.Frame):
    def __init__(self, parent, path=None):
        super().__init__(parent)
        self.path = path
        self.time_units = "s"
        self.tracking_method = "Optitrack"
        self.track_frequency = 120.0
        self.alignment_csv = "global"  # local or ttl
        self.csv_files = []

        self.align_to_headers = {
            "global": ["\n\n\n", "CSV files"],
            "local": ["\n\n\n", "CSV files", "Start epoch"],
            "ttl": (
                "\n\n\n",
                "CSV files",
                "TTL file",
                "Number\nof\nchannels",
                "Tracking\nchannel",
                "Bytes\nsize",
                "TTL\nsampling\nfrequency\n(Hz)",
                "Start\nepoch",
                "TTL\nthreshold",
            ),
        }

        self.parameters = pd.DataFrame(columns=["csv"])

        self.ttl_param_widgets = {}

        # Select tracking system
        fr1 = tk.Frame(master=self)
        fr1.pack(fill="x", padx=20, pady=5)
        tk.Label(master=fr1, text="Tracking system:").pack(side=tk.LEFT)
        self.tracking_method = tk.StringVar(fr1)
        self.tracking_method.set("Optitrack")
        self.tracking_menu = tk.OptionMenu(
            fr1,
            self.tracking_method,
            "Optitrack",
            "DeepLabCut",
            "Default",
        )
        self.tracking_menu.pack(side=tk.LEFT, padx=20)

        # Select type of alignment
        tk.Label(master=fr1, text="Tracking alignment:").pack(side=tk.LEFT, padx=20)
        texts = ["Global timestamps in CSV", "Local timestamps in CSV", "TTL detection"]
        values = ["global", "local", "ttl"]
        fr1rb = tk.Frame(master=fr1)
        fr1rb.pack(side=tk.LEFT)
        self.tk_alignment = tk.StringVar(None, self.alignment_csv)
        for i in range(3):
            tk.Radiobutton(
                master=fr1rb,
                text=texts[i],
                padx=5,
                pady=10,
                variable=self.tk_alignment,
                value=values[i],
                command=self._update_table_headers,
            ).pack(anchor="w")

        fr12 = tk.Frame(master=fr1)
        fr12.pack(side=tk.LEFT, fill="y", padx=20, pady=5)

        # Select tracking frequency
        tk.Label(master=fr12, text="Tracking frequency (Hz):").pack(
            padx=20
        )  # side=tk.LEFT, padx = 20)
        self.track_frequency = tk.DoubleVar(fr1)
        self.track_frequency.set(120.00)
        self.fr = tk.Entry(master=fr12, textvariable=self.track_frequency)
        self.fr.pack(padx=20, pady=20)  # side=tk.LEFT, padx = 5, expand=True)
        self.fr.bind("<Return>", lambda event: self.focus())

        # Load a CSV
        tk.Button(
            master=fr12, text="Load csv file(s)", command=self._load_csv_files
        ).pack()  # side=tk.LEFT, padx=20)

        # Initiate all tables
        self.tables = {}
        for k in ["global", "local", "ttl"]:
            self.tables[k] = TrackingTable(
                self, tuple(self.align_to_headers[k]), self.path, self.get_parameters(k)
            )

        self.tables["global"].pack(fill="both", padx=20, pady=5, expand=True)

        self.n_row = 0

    def get_parameters(self, alignment_csv):
        if alignment_csv == "global":
            return pd.DataFrame(columns=["csv"])
        elif alignment_csv == "local":
            return pd.DataFrame(columns=["csv", "epoch"])
        elif alignment_csv == "ttl":
            return pd.DataFrame(
                columns=[
                    "csv",
                    "ttl",
                    "n_channels",
                    "tracking_channel",
                    "bytes_size",
                    "fs",
                    "epoch",
                    "threshold",
                ]
            )

    def update_path_info(self, path):
        self.path = path

    def _update_table_headers(self):
        self.tables[self.alignment_csv].pack_forget()
        for item in self.tables[self.alignment_csv].get_children():
            self.tables[self.alignment_csv].delete(item)
        self.tables[self.alignment_csv].parameters = self.get_parameters(
            self.alignment_csv
        )
        self.alignment_csv = self.tk_alignment.get()
        self.tables[self.alignment_csv].pack(fill="both", padx=20, pady=5, expand=True)

    def _load_csv_files(self):
        suggested_dir = self.path if self.path else os.getenv("HOME")
        paths = filedialog.askopenfilenames(
            initialdir=suggested_dir,
            filetypes=[("CSV File(s)", "*.csv")],
            multiple=True,
        )
        for item in self.tables[self.alignment_csv].get_children():
            self.tables[self.alignment_csv].delete(item)
        self.csv_files = paths
        self.tables[self.alignment_csv]._update_parameters(
            self.csv_files, self.alignment_csv
        )

    def retrieve_tracking_parameters(self):
        return self.tables[self.alignment_csv].parameters


class EpochsTable(ttk.Treeview):
    def __init__(self, parent, n_row, n_col, path):
        super().__init__(parent, column=("start", "end", "label"), show="headings")

        self.path = path
        self.n_col = n_col
        self.n_row = n_row

        self.column("# 1", anchor=tk.CENTER)
        self.heading("# 1", text="start")
        self.column("# 2", anchor=tk.CENTER)
        self.heading("# 2", text="end")
        self.column("# 3", anchor=tk.CENTER)
        self.heading("# 3", text="label")

        yscrollbar = ttk.Scrollbar(self, orient="vertical", command=self.yview)
        self.configure(yscrollcommand=yscrollbar.set)
        yscrollbar.pack(side="right", fill="y")

        self.epochs = pd.DataFrame(index=[], columns=["start", "end", "label"])

        self.bind("<Double-1>", lambda event: self.on_double_click(event))
        self.tag_configure("oddrow", background="gray66")
        self.tag_configure("evenrow", background="gray92")
        self.tags = ["evenrow", "oddrow"]

    def open_sheet(self):
        self.check_change = False
        suggested_dir = self.path if self.path else os.getenv("HOME")
        path = filedialog.askopenfilename(
            initialdir=suggested_dir, filetypes=[("CSV File", "*.csv")]
        )
        if len(path):
            for item in self.get_children():
                self.delete(item)
            self.n_row = 0
            self.epochs = pd.read_csv(path, header=None)
            if len(self.epochs.columns) == 3:
                self.epochs.columns = ["start", "end", "label"]
            elif len(self.epochs.columns) == 2:
                self.epochs.columns = ["start", "end"]
                self.epochs["label"] = ""
            for r in self.epochs.index:
                self.insert(
                    "",
                    "end",
                    iid=r,
                    values=tuple(self.epochs.loc[r]),
                    tags=(self.tags[r % 2]),
                )
                self.n_row += 1

        return

    def on_double_click(self, event):
        # close previous popups
        try:  # in case there was no previous popup
            self.entryPopup.destroy()
        except AttributeError:
            pass

        rowid = self.identify_row(event.y)
        column = self.identify_column(event.x)
        if not rowid:
            return

        x, y, width, height = self.bbox(rowid, column)
        pady = height // 2
        text = self.item(rowid, "values")[int(column[1:]) - 1]
        self.entryPopup = EntryPopup(
            self, rowid, int(column[1:]) - 1, text, self.epochs
        )
        self.entryPopup.place(x=x, y=y + pady, width=width, height=height, anchor="w")

        return

    def insert_row(self):
        self.insert(
            "",
            "end",
            iid=self.n_row,
            values=("", "", ""),
            tags=(self.tags[self.n_row % 2]),
        )
        self.epochs.loc[self.n_row] = np.nan
        self.n_row += 1


class EpochsTab(ttk.Frame):
    def __init__(self, parent, path=None):
        super().__init__(master=parent)

        self.path = path
        self.time_units = "s'"

        # Select time units
        fr1 = tk.Frame(master=self)
        fr1.pack()
        ttk.Label(master=fr1, text="Please select the time units : ").pack(side=tk.LEFT)
        # self.time_units = ['s', 'ms', 'us']
        self.time_units = "s"
        self.tk_tu = tk.StringVar(None, self.time_units)
        for i, tu in enumerate(["s", "ms", "us"]):
            tk.Radiobutton(
                master=fr1,
                text=tu,
                padx=5,
                pady=10,
                variable=self.tk_tu,
                value=tu,
                command=self._time_units,
            ).pack(side=tk.LEFT)

        # Table view
        self.table = EpochsTable(self, 2, 3, path)
        self.table.pack(fill="both", padx=20, pady=5, expand=True)

        # Add row
        fr2 = tk.Frame(master=self)
        tk.Button(master=fr2, text="Add row", command=self.add_row).pack(fill="x")

        # Load a CSV
        tk.Button(master=fr2, text="Load csv file", command=self.load_csv_file).pack(
            fill="x"
        )
        fr2.pack(fill="x", padx=20, pady=10)

    def load_csv_file(self):
        self.table.open_sheet()

    def update_path_info(self, path):
        self.path = path
        # self.table.update_path_info(path)

    def _time_units(self):
        self.time_units = self.tk_tu.get()

    def add_row(self):
        self.table.insert_row()


class SessionInformationTab(ttk.Frame):
    def __init__(self, parent, path=None):
        super().__init__(parent)
        try:
            experimenter = getpass.getuser()
        except RuntimeError:
            experimenter = ""

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        session_frame = tk.Frame(master=self, padx=15, pady=10)
        session_frame.columnconfigure(0, weight=1)
        session_frame.columnconfigure(1, weight=5)
        session_frame.rowconfigure(1)
        session_frame.grid(row=0, column=0, sticky="nswe")
        ttk.Label(master=session_frame, text="Session").grid(row=0, column=0)
        ttk.Label(master=session_frame, text="_______").grid(row=1, column=0)

        self.session_information = {
            "path": tk.StringVar(master=session_frame, value=path),
            "name": tk.StringVar(master=session_frame),
            "description": tk.StringVar(master=session_frame),
            "experimenter": tk.StringVar(master=session_frame, value=experimenter),
            "lab": tk.StringVar(master=session_frame),
            "institution": tk.StringVar(master=session_frame),
        }

        for i, n in enumerate(self.session_information.keys()):
            if n != "path":
                tk.Label(master=session_frame, text=n).grid(
                    row=i + 2, column=0, sticky="W"
                )
                ent = tk.Entry(
                    master=session_frame, textvariable=self.session_information[n]
                )
                ent.grid(row=i + 2, column=1, sticky="EW", columnspan=4)
                if n == "name":
                    self.name = self.session_information["name"]

        subject_frame = tk.Frame(master=self, padx=15, pady=10)
        subject_frame.columnconfigure(0, weight=1)
        subject_frame.columnconfigure(1, weight=5)
        subject_frame.rowconfigure(1)
        subject_frame.grid(row=0, column=1, sticky="nswe")
        ttk.Label(master=subject_frame, text="Subject").grid(row=0, column=0)
        ttk.Label(master=subject_frame, text="_______").grid(row=1, column=0)

        self.subject_information = {
            "age": tk.StringVar(master=subject_frame),
            "description": tk.StringVar(master=subject_frame),
            "genotype": tk.StringVar(master=subject_frame),
            "sex": tk.StringVar(master=subject_frame),
            "species": tk.StringVar(master=subject_frame),
            "subject_id": tk.StringVar(master=subject_frame),
            "weight": tk.StringVar(master=subject_frame),
            # 'date_of_birth':'',
            "strain": tk.StringVar(master=subject_frame),
        }

        for i, n in enumerate(self.subject_information.keys()):
            tk.Label(master=subject_frame, text=n).grid(row=i + 2, column=0, sticky="W")
            ent = tk.Entry(
                master=subject_frame, textvariable=self.subject_information[n]
            )
            ent.grid(row=i + 2, column=1, sticky="EW", columnspan=4)

        self.update_path_info(path)

    def retrieve_session_information(self):
        for n in self.session_information.keys():
            self.session_information[n] = self.session_information[n].get()
        return self.session_information

    def retrieve_subject_information(self):
        for n in self.subject_information.keys():
            self.subject_information[n] = self.subject_information[n].get()
        return self.subject_information

    def update_path_info(self, path):
        self.session_information["path"].set(path)
        self.session_information["name"].set(os.path.basename(path) if path else "")
        # self.name.set(self.session_information["name"])


class BaseLoaderGUI(ttk.Frame):
    def __init__(self, container, path=""):
        super().__init__(container)
        self.container = container

        # Basic properties to return
        self.status = False
        self.path = path

        topframe = tk.Frame()
        topframe.pack(fill="x", pady=10)
        tk.Label(master=topframe, text="Pynapple").pack()
        tk.Label(master=topframe, text="Data directory").pack(side=tk.LEFT, padx=10)
        self.directory_line = tk.StringVar()
        tk.Entry(master=topframe, text=self.directory_line).pack(
            side=tk.LEFT, expand=True, fill="x", padx=10
        )
        self.directory_line.set(path)
        tk.Button(master=topframe, text="Browse", command=self.select_folder).pack(
            side=tk.LEFT, padx=20
        )

        nb = ttk.Notebook()
        self.tab_session = SessionInformationTab(nb, self.path)
        nb.add(self.tab_session, text="Session Information")
        self.tab_epoch = EpochsTab(nb, self.path)
        nb.add(self.tab_epoch, text="Epochs")
        self.tab_tracking = TrackingTab(nb, self.path)
        nb.add(self.tab_tracking, text="Tracking")

        nb.pack(expand=1, fill="both")

        botframe = tk.Frame()
        botframe.pack(fill="x", padx=20)

        tk.Button(master=botframe, text="Ok", command=self.accept).pack(
            pady=10, side=tk.RIGHT
        )
        tk.Button(master=botframe, text="Cancel", command=self.reject).pack(
            pady=10, side=tk.RIGHT
        )

    def accept(self):
        self.status = True
        # # Collect all the information acquired
        self.session_information = self.tab_session.retrieve_session_information()
        self.subject_information = self.tab_session.retrieve_subject_information()
        self.epochs = self.tab_epoch.table.epochs
        self.time_units_epochs = self.tab_epoch.time_units
        self.tracking_parameters = self.tab_tracking.retrieve_tracking_parameters()
        self.tracking_alignment = self.tab_tracking.alignment_csv
        self.tracking_method = self.tab_tracking.tracking_method.get()
        self.tracking_frequency = self.tab_tracking.track_frequency.get()
        self.container.destroy()

    def reject(self):
        self.status = False
        self.container.destroy()

    def select_folder(self):
        self.path = filedialog.askdirectory()
        self.directory_line.set(self.path)
        self.tab_session.update_path_info(self.path)
        self.tab_epoch.update_path_info(self.path)
        self.tab_tracking.update_path_info(self.path)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("The minimalist session loader")
        self.geometry("1000x550")
