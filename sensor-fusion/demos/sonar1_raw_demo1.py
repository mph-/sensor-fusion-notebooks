# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show


def sonar1_raw_demo1_plot(axes):

    raw_max = 4

    # Load data
    filename = 'data/sonar1-calibration.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Split into columns
    distance, sonar1 = data.T

    axes.clear()
    axes.plot(distance, sonar1, '.', alpha=0.2)
    axes.set_xlabel('Distance (m)')
    axes.set_ylabel('Measurement (m)')
    axes.set_ylim(0, raw_max)


def sonar1_raw_demo1():

    fig, axes = subplots(1)
    show()

    interact(sonar1_raw_demo1_plot, axes=fixed(axes),
             continuous_update=False)
