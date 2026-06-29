# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed, interactive, fixed
from matplotlib.pyplot import subplots, show
from .lib.signal_plot import create_axes

def sonar1_model_demo1_plot(N=20, order=2, ignore_outliers=True):

    # Load data
    filename = 'data/sonar1-calibration.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Split into columns
    distance, sonar1 = data.T
    
    dmin = distance.min()
    dmax = distance.max()
    
    dr = (dmax - dmin) / N
    r = (np.arange(N) + 0.5) * dr
    
    error = sonar1 - distance

    means = np.zeros(N)
    stds = np.zeros(N)
    
    for n in range(N):
        dmin = r[n] - 0.5 * dr
        dmax = r[n] + 0.5 * dr
        if ignore_outliers:
            m = (distance > dmin) & (distance < dmax) & (abs(error) < 0.2)
        else:
            m = (distance > dmin) & (distance < dmax)

        means[n] = error[m].mean()
        stds[n] = error[m].std()

    mean_model = np.polyval(np.polyfit(r, means, order), r)
    std_model = np.polyval(np.polyfit(r, stds, order), r)        
        
    axes, kwargs = create_axes(2)
    axes[0].plot(r, means, '.')
    axes[0].plot(r, mean_model)
    axes[0].set_xlabel('Distance (m)')
    axes[0].set_ylabel('Error mean (m)')
    axes[1].plot(r, stds, '.')
    axes[1].plot(r, std_model)    
    axes[1].set_xlabel('Distance (m)')
    axes[1].set_ylabel('Error std dev (m)')
    show()

def sonar1_model_demo1():
    interact(sonar1_model_demo1_plot, N=(1, 50), order=(1, 5),
             continuous_update=False)
