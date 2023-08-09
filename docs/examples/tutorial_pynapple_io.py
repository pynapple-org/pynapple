# coding: utf-8

"""
IO Tutorial
===========

This notebook is designed to demonstrate the pynapple IO. It is build around the specifications of the [BIDS standard](https://bids-standard.github.io/bids-starter-kit/index.html) for sharing datasets. The key ideas are summarized as follow :

- [Hierarchy of folders](https://bids-standard.github.io/bids-starter-kit/folders_and_files/folders.html)
    
    ![Image title](../../_static/BIDS_Folders.png){ align=left }
    
- [Filename template](https://bids-standard.github.io/bids-starter-kit/folders_and_files/files.html)

    ![Image title](../../_static/BIDS_Files.png){ align=left }

- [Metadata files](https://bids-standard.github.io/bids-starter-kit/folders_and_files/metadata.html)
    
    ![Image title](../../_static/BIDS_Metadata.png){ align=left }

"""

# %%
# ***
# Navigating a structured dataset
# -------------------------------
#
# The dataset in this example can be found [here](https://www.dropbox.com/s/pr1ze1nuiwk8kw9/MyProject.zip?dl=1).


import numpy as np
import pynapple as nap

# mkdocs_gallery_thumbnail_path = '_static/treeview.png'

project_path = "../../your/path/to/MyProject"

project = nap.load_folder(project_path)

print(project)

# %%
# The pynapple IO offers a convenient way of visualizing and navigating a folder based dataset. To visualize the whole hierarchy of Folders, you can call the view property or the expand function.

project.view

# %%
# Here it shows all the subjects (in this case only A2929), all the sessions and all of the derivatives folders. It shows as well all the NPZ files that contains a pynapple object and the NWB files.
#
# The object project behaves like a nested dictionnary. It is then easy to loop and navigate through a hierarchy of folders when doing analyses. In this case, we are gonna take only the session A2929-200711.


session = project["sub-A2929"]["ses-A2929-200711"]

print(session)

# %%
# I can expand to see what the folders contains.

print(session.expand())


# %%
# ***
# Loading files
# -------------
#
# By default, pynapple save objects as NPZ. It is a convenient way to save all the properties of an object such as the time support. The pynapple IO offers an easy way to load any NPZ files that matches the structures defined for a pynapple object.

spikes = session["derivatives"]["spikes"]
position = session["derivatives"]["position"]
wake_ep = session["derivatives"]["wake_ep"]
sleep_ep = session["derivatives"]["sleep_ep"]

# %%
# Objects are only loaded when they are called.

print(session["derivatives"]["spikes"])

# %%
# ***
# Metadata
# --------
#
# A good practice for sharing datasets is to write as many metainformation as possible. Following BIDS specifications, any data files should be accompagned by a JSON sidecar file.

import os

for f in os.listdir(session["derivatives"].path):
    print(f)

# %%
# To read the metainformation associated with a file, you can use the functions `doc`, `info` or `metadata` :

session["derivatives"].doc("spikes")


session["derivatives"].doc("position")

# %%
# ***
# Saving a pynapple object
# ------------------------
#
# In this case, we define a new Tsd and a new IntervalSet that we would like to save in the session folder.

tsd = position["x"] + position["y"]
epoch = nap.IntervalSet(start=np.array([0, 3]), end=np.array([1, 6]))

session.save("x_plus_y", tsd, description="Random position")
session.save("stimulus-fish", epoch, description="Fish pictures to V1")

# %%
# We can visualize the newly saved objects.

session.expand()

# %%
session.doc("stimulus-fish")

# %%
session["x_plus_y"]
