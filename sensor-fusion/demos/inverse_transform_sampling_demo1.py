# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from matplotlib.pyplot import subplots, show
from scipy.interpolate import interp1d
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


def inverse_transform_sampling_demo1_plot(axes, distX=distributions[0],
                                          muX=0, sigmaX=1, u=0.5):

    Nx = 801
    x = np.linspace(-10, 10, Nx)
    dx = x[1] - x[0]

    fX = pdf(x, muX, sigmaX, distX)
    FX = np.cumsum(fX) * dx

    interp = interp1d(FX, x, kind='linear', bounds_error=False,
                      fill_value=x[-1])
    x1 = interp(u)

    axes[0].clear()
    axes[0].plot(x, fX, label='PDF')
    axes[0].set_ylim(0, 0.5)
    axes[0].legend()
    axes[0].set_xlim(-5, 5)

    axes[1].clear()
    axes[1].plot(x, FX, color='orange', label='CDF')
    axes[1].plot((-5, x1), (u, u), color='green')
    axes[1].plot((x1, x1), (-0.1, u), color='green')
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()
    axes[1].set_xlim(-5, 5)


def inverse_transform_sampling_demo1():

    fig, axes = subplots(2)
    show()

    interact(inverse_transform_sampling_demo1_plot, axes=fixed(axes),
             distX=distributions, muX=(-2, 2), sigmaX=(0.01, 5, 0.01),
             u=(0, 1, 0.05), continuous_update=False)
