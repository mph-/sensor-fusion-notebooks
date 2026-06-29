# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from matplotlib.pyplot import subplots, show
from ipywidgets import interact, fixed
from .lib.utils import gauss

distributions = ['gaussian', 'uniform']


def pdf(x, muX, sigmaX, distribution):

    if distribution == 'gaussian':
        return gauss(x, muX, sigmaX)
    elif distribution == 'uniform':
        xmin = muX - np.sqrt(12) * sigmaX / 2
        xmax = muX + np.sqrt(12) * sigmaX / 2
        return 1.0 * ((x >= xmin) & (x <= xmax)) / (xmax - xmin)
    raise ValueError('Unknown distribution %s' % distribution)


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def cdf_demo1_plot(axes, distX=distributions[0], muX=0, sigmaX=1, x=-0.5):

    x1 = x
    Nx = 801
    x = np.linspace(-10, 10, Nx)
    dx = x[1] - x[0]

    fX = pdf(x, muX, sigmaX, distX)
    FX = np.cumsum(fX) * dx

    m1 = find_nearest_idx(x, x1)
    FX1 = FX[m1]

    axes[0].clear()
    axes[0].plot(x, fX, label='PDF')
    axes[0].fill_between(x[x < x1], 0, fX[x < x1], facecolor='none',
                         edgecolor='b', hatch='///')
    axes[0].set_ylim(0, 0.5)

    axes[1].clear()
    axes[1].plot(x, FX, color='orange', label='CDF')
    axes[1].plot(x1, FX1, 'o', color='orange')
    axes[0].legend()
    axes[1].legend()
    axes[0].set_xlim(-5, 5)
    axes[1].set_xlim(-5, 5)


def cdf_demo1():

    fig, axes = subplots(2)
    show()

    interact(cdf_demo1_plot, axes=fixed(axes),
             distX=distributions,
             muX=(-2, 2), sigmaX=(0.01, 5, 0.01),
             x=(-5, 5, 0.1),  continuous_update=False)
