# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show


def sonar1_fit_hist_demo1_plot(axes, distance=1.0, width=0.5, error_max=0.2,
                               ignore_outliers=True):

    # Load data
    filename = 'data/sonar1-calibration.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Split into columns
    distance1, sonar1 = data.T

    dmin = distance - 0.5 * width
    dmax = distance + 0.5 * width

    if ignore_outliers:
        m = abs(distance1 - sonar1) < 0.1
        distance1 = distance1[m]
        sonar1 = sonar1[m]

    p = np.polyfit(distance1, sonar1, 1)
    error = np.polyval(p, distance1) - sonar1

    m = (distance1 < dmax) & (distance1 > dmin)

    bins = np.linspace(-error_max, error_max, 100)

    smean = error[m].mean()
    sstd = error[m].std()

    axes[0].clear()
    axes[0].plot(distance1, error, '.', alpha=0.2)
    axes[0].set_xlabel('Distance (m)')
    axes[0].set_ylabel('Error (m)')
    axes[0].set_ylim(-error_max, error_max)
    axes[0].axvspan(dmin, dmax, color='orange', alpha=0.4)
    axes[0].set_title('mean = %.2f  std = %.2f' % (smean, sstd))

    axes[1].clear()
    axes[1].hist(error[m], bins=bins, color='orange')
    axes[1].set_xlabel('Error (m)')
    axes[1].set_xlim(-error_max, error_max)


def sonar1_fit_hist_demo1():

    fig, axes = subplots(2)
    show()

    interact(sonar1_fit_hist_demo1_plot, axes=fixed(axes),
             distance=(0, 3.4, 0.2), width=(0.1, 1, 0.1),
             error_max=(0.05, 0.5, 0.05), continuous_update=False)
