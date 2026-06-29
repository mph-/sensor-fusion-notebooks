# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from numpy.linalg import lstsq


def model(r, k):
    # This model is not that good!
    return k[0] + k[1] / (r + k[2])


def model_nonlinear_least_squares_fit(r, v, iterations=5):

    N = len(r)
    A = np.ones((N, 3))
    k = np.zeros(3)

    for i in range(iterations):
        # Calculate Jacobians for current estimate of parameters.
        for n in range(N):
            A[n, 1] = 1 / (r[n] + k[2])
            A[n, 2] = -k[1] / (r[n] + k[2])**2

        # Use least squares to estimate the parameters.
        deltak, res, rank, s = lstsq(A, v - model(r, k), rcond=None)
        k += deltak
    return k


def ir3_hist_demo1_plot(axes, distance=1.0, width=0.5):

    # Load data
    filename = 'data/ir3-calibration.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Split into columns
    distance1, ir3 = data.T

    data = ir3

    k = model_nonlinear_least_squares_fit(distance1, data)
    data_fit = model(distance1, k)

    error = data - data_fit

    error_max = 0.2
    zmin = -error_max
    zmax = error_max

    dmin = distance - 0.5 * width
    dmax = distance + 0.5 * width

    m = (distance1 < dmax) & (distance1 > dmin)

    bins = np.linspace(zmin, zmax, 100)

    smean = error[m].mean()
    sstd = error[m].std()

    axes[0].clear()
    axes[0].plot(distance1, data, '.', alpha=0.2)
    axes[0].plot(distance1, data_fit)
    axes[0].set_xlabel('Distance (m)')
    axes[0].set_ylabel('Measurement')
    axes[0].set_ylim(0, 4)
    axes[0].axvspan(dmin, dmax, color='C1', alpha=0.4)
    axes[0].set_title('mean = %.2f  std = %.2f' % (smean, sstd))

    axes[1].clear()
    axes[1].plot(distance1, data - data_fit, '.', alpha=0.2)
    axes[1].set_xlabel('Distance (m)')
    axes[1].set_ylabel('Error')
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].axvspan(dmin, dmax, color='C1', alpha=0.4)

    axes[2].clear()
    axes[2].hist(error[m], bins=bins, color='C2')
    axes[2].set_xlabel('Error')
    axes[2].set_ylabel('Count')
    axes[2].set_xlim(zmin, zmax)


def ir3_hist_demo1():

    fig, axes = subplots(3)
    show()

    interact(ir3_hist_demo1_plot, axes=fixed(axes), distance=(0, 3.5, 0.2),
             width=(0.1, 1, 0.1), continuous_update=False)
