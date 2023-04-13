# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-03-25 11:34:45
# @Last Modified by:   gviejo
# @Last Modified time: 2023-04-13 13:34:56


import tkinter as tk
from tkinter import ttk


class OphysGUI(ttk.Frame):
    def __init__(self, container, path=""):
        super().__init__(container)
        self.container = container

        # Basic properties to return
        self.status = False
        self.path = path
        self.ophys = {
            "device": {
                n: d
                for n, d in zip(
                    ["name", "description", "manufacturer"],
                    [
                        tk.StringVar(self, value="Microscope"),
                        tk.StringVar(self),
                        tk.StringVar(self),
                    ],
                )
            },
            "OpticalChannel": {
                n: d
                for n, d in zip(
                    ["name", "description", "emission_lambda"],
                    [
                        tk.StringVar(self, value="OpticalChannel"),
                        tk.StringVar(self),
                        tk.StringVar(self, "500."),
                    ],
                )
            },
            "ImagingPlane": {
                n: d
                for n, d in zip(
                    [
                        "name",
                        "imaging_rate",
                        "description",
                        "excitation_lambda",
                        "indicator",
                        "location",
                    ],
                    [
                        tk.StringVar(self, value="ImagingPlane"),
                        tk.StringVar(self, value="30."),
                        tk.StringVar(self),
                        tk.StringVar(self, value="600."),
                        tk.StringVar(self, value="GCAMP"),
                        tk.StringVar(self),
                    ],
                )
            },
            "PlaneSegmentation": {
                n: d
                for n, d in zip(
                    ["name", "description"],
                    [tk.StringVar(self, "PlaneSegmentation"), tk.StringVar(self)],
                )
            },
        }

        topframe = tk.Frame()
        topframe.pack(fill="x", pady=10)
        tk.Label(master=topframe, text="Pynapple").pack()
        tk.Label(master=topframe, text=path).pack()

        frame2 = tk.Frame()
        frame2.pack(fill="both", pady=10)
        frame2.columnconfigure(3)

        n_row = 0
        for i, n in enumerate(self.ophys.keys()):
            tk.Label(master=frame2, text=n).grid(row=n_row, column=0, sticky="nswe")
            n_row += 1
            for k in self.ophys[n].keys():
                tk.Label(master=frame2, text=k).grid(row=n_row, column=1, sticky="nswe")
                entry = tk.Entry(master=frame2, textvariable=self.ophys[n][k], width=25)
                entry.grid(row=n_row, column=2)  # , sticky="nswe")
                entry.bind("<Return>", self.on_return)
                n_row += 1
            n_row += 1

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
        self.ophys_information = {}
        for i, n in enumerate(self.ophys.keys()):
            self.ophys_information[n] = {}
            for k in self.ophys[n].keys():
                self.ophys_information[n][k] = self.ophys[n][k].get()

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
        self.title("Calcium Imaging Loader")
        self.geometry("650x660")
