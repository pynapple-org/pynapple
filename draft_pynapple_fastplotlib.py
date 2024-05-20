# -*- coding: utf-8 -*-
"""
Fastplotlib
===========

Working with calcium data.

For the example dataset, we will be working with a recording of a freely-moving mouse imaged with a Miniscope (1-photon imaging). The area recorded for this experiment is the postsubiculum - a region that is known to contain head-direction cells, or cells that fire when the animal's head is pointing in a specific direction.

The NWB file for the example is hosted on [OSF](https://osf.io/sbnaw). We show below how to stream it.

See the [documentation](https://pynapple-org.github.io/pynapple/) of Pynapple for instructions on installing the package.

This tutorial was made by Sofia Skromne Carrasco and Guillaume Viejo.

"""
# %%
# %gui qt

import pynapple as nap
import numpy as np
import fastplotlib as fpl

import sys
# mkdocs_gallery_thumbnail_path = '../_static/fastplotlib_demo.png'

def get_memory_map(filepath, nChannels, frequency=20000):
    n_channels = int(nChannels)
    f = open(filepath, 'rb') 
    startoffile = f.seek(0, 0)
    endoffile = f.seek(0, 2)
    bytes_size = 2      
    n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
    duration = n_samples/frequency
    interval = 1/frequency
    f.close()
    fp = np.memmap(filepath, np.int16, 'r', shape = (n_samples, n_channels))        
    timestep = np.arange(0, n_samples)/frequency

    return fp, timestep


#### LFP
data_array, time_array = get_memory_map("your/path/to/MyProject/sub-A2929/A2929-200711/A2929-200711.dat", 16)
lfp = nap.TsdFrame(t=time_array, d=data_array)

lfp2 = lfp.get(0, 20)[:,14]
lfp2 = np.vstack((lfp2.t, lfp2.d)).T

#### NWB
nwb = nap.load_file("your/path/to/MyProject/sub-A2929/A2929-200711/pynapplenwb/A2929-200711.nwb")
units = nwb['units']#.getby_category("location")['adn']
tmp = units.to_tsd().get(0, 20)
tmp = np.vstack((tmp.index.values, tmp.values)).T 



fig = fpl.Figure(canvas="glfw", shape=(2,1))
fig[0,0].add_line(data=lfp2, thickness=1, cmap="autumn")
fig[1,0].add_scatter(tmp)
fig.show(maintain_aspect=False)
# fpl.run()




# grid_plot = fpl.GridPlot(shape=(2, 1), controller_ids="sync", names = ['lfp', 'wavelet'])
# grid_plot['lfp'].add_line(lfp.t, lfp[:,14].d)


import numpy as np
import fastplotlib as fpl

fig = fpl.Figure(canvas="glfw")#, shape=(2,1), controller_ids="sync")
fig[0,0].add_line(data=np.random.randn(1000))
fig.show(maintain_aspect=False)

fig2 = fpl.Figure(canvas="glfw", controllers=fig.controllers)#, shape=(2,1), controller_ids="sync")
fig2[0,0].add_line(data=np.random.randn(1000)*1000)
fig2.show(maintain_aspect=False)



# Not sure about this :
fig[1,0].controller.controls["mouse1"] = "pan", "drag", (1.0, 0.0)

fig[1,0].controller.controls.pop("mouse2")
fig[1,0].controller.controls.pop("mouse4")
fig[1,0].controller.controls.pop("wheel")

import pygfx

controller = pygfx.PanZoomController()
controller.controls.pop("mouse1")
controller.add_camera(fig[0, 0].camera)
controller.register_events(fig[0, 0].viewport)

controller2 = pygfx.PanZoomController()
controller2.add_camera(fig[1, 0].camera)
controller2.controls.pop("mouse1")
controller2.register_events(fig[1, 0].viewport)
















sys.exit()

#################################################################################################


nwb = nap.load_file("your/path/to/MyProject/sub-A2929/A2929-200711/pynapplenwb/A2929-200711.nwb")
units = nwb['units']#.getby_category("location")['adn']
tmp = units.to_tsd()
tmp = np.vstack((tmp.index.values, tmp.values)).T 

# Example 1

fplot = fpl.Plot()
fplot.add_scatter(tmp)
fplot.graphics[0].cmap = "jet" 
fplot.graphics[0].cmap.values = tmp[:, 1]
fplot.show(maintain_aspect=False)

# Example 2

names = [['raster'], ['position']]
grid_plot = fpl.GridPlot(shape=(2, 1), controller_ids="sync", names = names)
grid_plot['raster'].add_scatter(tmp)
grid_plot['position'].add_line(np.vstack((nwb['ry'].t, nwb['ry'].d)).T)
grid_plot.show(maintain_aspect=False)
grid_plot['raster'].auto_scale(maintain_aspect=False)


# Example 3
#frames = iio.imread("/Users/gviejo/pynapple/A0670-221213_filtered.avi")
#frames = frames[:,:,:,0]
frames = np.random.randn(10, 100, 100)

iw = fpl.ImageWidget(frames, cmap="gnuplot2")

#iw.show()

# Example 4 

from PyQt6 import QtWidgets


mainwidget = QtWidgets.QWidget()

hlayout = QtWidgets.QHBoxLayout(mainwidget)

iw.widget.setParent(mainwidget)

hlayout.addWidget(iw.widget)

grid_plot.widget.setParent(mainwidget)

hlayout.addWidget(grid_plot.widget)

mainwidget.show()
