# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed, interactive, fixed
from matplotlib.pyplot import subplots, show
from .lib.signal_plot import create_axes

def sonar1_hist_demo1_plot(distance=1.0, width=0.5, error_max=0.2,
                           ignore_outliers=True):

    # Load data
    filename = 'data/sonar1-calibration.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Split into columns
    distance1, sonar1 = data.T
    
    error = sonar1 - distance1

    dmin = distance - 0.5 * width
    dmax = distance + 0.5 * width    
    
    m = (distance1 < dmax) & (distance1 > dmin)
    if ignore_outliers:
        m = m & (abs(error) < 0.2)

    bins = np.linspace(-error_max, error_max, 100)

    smean = error[m].mean()
    sstd = error[m].std()    
    
    axes, kwargs = create_axes(2)
    axes[0].plot(distance1, error, '.', alpha=0.2)
    axes[0].set_xlabel('Distance (m)')
    axes[0].set_ylabel('Error (m)')
    axes[0].set_ylim(-error_max, error_max)
    axes[0].axvspan(dmin, dmax, color='orange', alpha=0.4)
    axes[0].set_title('mean = %.2f  std = %.2f' % (smean, sstd))
    
    axes[1].hist(error[m], bins=bins, color='orange')
    axes[1].set_xlabel('Error (m)')
    axes[1].set_xlim(-error_max, error_max)    

def sonar1_hist_demo1():
    interact(sonar1_hist_demo1_plot, distance=(0, 3.5, 0.2),
             width=(0.1, 1, 0.1), error_max=(0.05, 0.5, 0.05),
             continuous_update=False)
