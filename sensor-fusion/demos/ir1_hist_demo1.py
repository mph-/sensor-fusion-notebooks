# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact
from matplotlib.pyplot import show
from .lib.signal_plot import create_axes


def ir1_hist_demo1_plot(distance=1.0, width=0.5, zmin=0, zmax=4,
                        ignore_outliers=True):

    # Load data
    filename = 'data/ir1-calibration.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Split into columns
    distance1, ir1 = data.T

    data = ir1

    zmin = 0
    zmax = 4

    dmin = distance - 0.5 * width
    dmax = distance + 0.5 * width

    m = (distance1 < dmax) & (distance1 > dmin)

    bins = np.linspace(zmin, zmax, 100)

    smean = data[m].mean()
    sstd = data[m].std()

    axes, kwargs = create_axes(2)
    axes[0].plot(distance1, data, '.', alpha=0.2)
    axes[0].set_xlabel('Distance (m)')
    axes[0].set_ylabel('Measurement')
    axes[0].set_ylim(zmin, zmax)
    axes[0].axvspan(dmin, dmax, color='orange', alpha=0.4)
    axes[0].set_title('mean = %.2f  std = %.2f' % (smean, sstd))

    axes[1].hist(data[m], bins=bins, color='orange')
    axes[1].set_xlabel('Measurement')
    axes[1].set_xlim(zmin, zmax)

    show()


def ir1_hist_demo1():
    interact(ir1_hist_demo1_plot, distance=(0, 3.5, 0.2),
             width=(0.1, 1, 0.1), zmin=(0.1, 4, 0.1), zmax=(0.1, 4, 0.1),
             continuous_update=False)
