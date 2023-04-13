"""Summary
"""
import tkinter as tk
from tkinter import ttk

import numpy as np


class EphysGUI(ttk.Frame):
    def __init__(self, container, path="", groups={}):
        super().__init__(container)
        self.container = container

        # Basic properties to return
        self.status = False
        self.path = path
        self.groups = groups
        self.ephys = {}
        for k in groups.keys():
            self.ephys[k] = {}
            self.ephys[k]["electrodes"] = " ".join(groups[k][0:9].astype(np.str_))
            if len(groups[k]) > 9:
                self.ephys[k]["electrodes"] += " ..."
            for n in [
                "name",
                "description",
                "location",
                "position",
                "device.name",
                "device.description",
                "device.manufacturer",
            ]:
                self.ephys[k][n] = tk.StringVar(self)

        topframe = tk.Frame()
        topframe.pack(fill="x", pady=10)
        tk.Label(master=topframe, text="Pynapple").pack()
        tk.Label(master=topframe, text=path).pack()

        midframe = tk.Frame()
        midframe.pack(fill="both", pady=10)
        # midframe.columnconfigure(3)

        self.canvas = tk.Canvas(midframe, height=720, width=600)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        frame2 = tk.Frame(self.canvas)
        self.canvas.create_window(0, 0, window=frame2, anchor="nw")

        self.entries = {}

        n_row = 0
        for i, n in enumerate(self.ephys.keys()):
            tk.Label(master=frame2, text="Group " + str(n)).grid(
                row=n_row, column=0, sticky="nswe"
            )
            n_row += 1
            self.entries[n] = {}
            for k in self.ephys[n].keys():
                tk.Label(master=frame2, text=k).grid(row=n_row, column=1, sticky="nswe")
                if k == "electrodes":
                    tk.Label(master=frame2, text=self.ephys[n][k]).grid(
                        row=n_row, column=2, sticky="nswe"
                    )
                else:
                    self.entries[n][k] = tk.Entry(
                        master=frame2, textvariable=self.ephys[n][k], width=25
                    )
                    self.entries[n][k].grid(row=n_row, column=2)  # , sticky="nswe")
                    # self.entries[n][k].bind('<Return>', lambda event: self.focus())
                    self.entries[n][k].bind("<Return>", self.on_return)
                n_row += 1
            n_row += 1

        photoScroll = tk.Scrollbar(midframe, orient=tk.VERTICAL)
        photoScroll.config(command=self.canvas.yview)
        self.canvas.config(yscrollcommand=photoScroll.set)
        photoScroll.grid(row=0, column=1, sticky="ns")
        midframe.bind("<Configure>", self.update_scrollregion)

        botframe = tk.Frame()
        botframe.pack(fill="x", padx=20)

        tk.Button(master=botframe, text="Ok", command=self.accept).pack(
            pady=10, side=tk.RIGHT
        )
        tk.Button(master=botframe, text="Cancel", command=self.reject).pack(
            pady=10, side=tk.RIGHT
        )

    def update_scrollregion(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def accept(self):
        # Retrieve everything
        self.ephys_information = {}
        for i, n in enumerate(self.ephys.keys()):
            self.ephys_information[n] = {}
            device = {}
            for k in self.ephys[n].keys():
                if isinstance(self.ephys[n][k], tk.StringVar):
                    if "device" in k:
                        device[k.split(".")[1]] = self.ephys[n][k].get()
                    else:
                        self.ephys_information[n][k] = self.ephys[n][k].get()
                else:
                    self.ephys_information[n][k] = self.ephys[n][k]
            self.ephys_information[n]["device"] = device

        self.container.destroy()

        self.status = True
        return self.status

    def reject(self):
        self.status = False
        self.container.destroy()

    def on_return(self, event):
        self.container.focus()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ephys Loader")
        self.geometry("650x900")
