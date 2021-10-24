#!/usr/bin/env python

'''
	File name: main1.py
	Author: Guillaume Viejo
	Date created: 12/10/2017    
	Python Version: 3.5.2

First let's import the package and verify that everything is installed
You should write each line in ipython3 yourself. That's a good way to learn
'''

# Numpy is for handling matrices and vector :
import numpy as np
# if no error, you can type to see for example the version :
print(np.__version__)
# Now let's declare a matrice of zeros of shape 3 by 3:
mymatrice1 = np.zeros((3,3))
mymatrice1
# you can now see the shape of your matrice by doing 
mymatrice1.shape
# let's declare a matrice of ones of shape 5 by 3
mymatrice2 = np.ones((5,3))
# We can change the top left value by indexing mymatrice2 and assigning a new value
mymatrice2[0,0] = 2.0
# IN PYTHON (AND IN ALL PROGRAMMING LANGUAGES EXCEPT MATLAB) INDEXING START AT 0 
# so the bottom right element will be 
mymatrice2[4,2] = 4.0 
# Observe the change by typing the name of your matrice
mymatrice2
# now try to change each value of your matrice independantly
# To declare a matrice of random values between 0 and 1 of shape 1 by 10
my_random_matrice = np.random.rand(10)
# To compute the average value of my random points
mean_value = np.mean(my_random_matrice)
# To compute the variance of my random points
var_value = np.var(my_random_matrice)


# Pandas is a cool package to handle big datasets
import pandas as pd
# let's put mymatrice2 in a pandas dataframe 
my_panda = pd.DataFrame(data = mymatrice2)
# observe the difference when calling the variable
my_panda
# pandas shows the value inside the matrice but also the index along the two dimensions of the matrice
# We can change the index (if you haven't change the shape of mymatrice2)
my_panda = pd.DataFrame(data = mymatrice2, index = ['Janvier', 'Fevrier', 'Mars', 'Avril', 'Mai'], columns = ['Yes', 'No', 'Maybe'])
# What's happening in Fevrier?
my_panda.loc['Fevrier']
# In Mai?
my_panda.loc['Mai']
# For Maybe?
my_panda['Maybe']
# No during Avril
my_panda.loc['Avril','No']
# Yes during Janvier and Mars
my_panda.loc[['Janvier','Mars'],'Yes']
# observe the use of .loc for line indexing and the use of the [] for multiple index

# Matplotlib to plot
import matplotlib.pyplot as plt
# let's plot a sinusoidal function 
# first we declare the phase value between 0 and 4pi with a 0.01 step
phi = np.arange(0, 4*np.pi, 0.01)
# let's declare a figure
plt.figure()
# and let's plot sin(phi)
plt.plot(phi, np.sin(phi), color = 'red')
# always label the axes
plt.xlabel("Phase")
plt.ylabel("Sinus")
# a title
plt.title("My plot")
# and display the figure
plt.show()
# no let's display a matrix
image = np.random.rand(20,30)
plt.imshow(image)
plt.show()
