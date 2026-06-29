# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed, interactive, fixed
from matplotlib.pyplot import subplots, show
from .lib.signal_plot import create_axes

def sonar1_error_demo1_plot(error_max=0.2):

    # Load data
    filename = 'data/sonar1-calibration.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Split into columns
    distance, sonar1 = data.T
    
    dmin = distance.min()
    dmax = distance.max()
    
    error = sonar1 - distance

    axes, kwargs = create_axes(1)
    axes.plot(distance, error, '.', alpha=0.2)
    axes.set_xlabel('Distance (m)')
    axes.set_ylabel('Error (m)')
    axes.set_ylim(-error_max, error_max)

def sonar1_error_demo1():
    interact(sonar1_error_demo1_plot, error_max=(0.05, 3, 0.05),
             continuous_update=False)
