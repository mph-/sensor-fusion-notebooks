# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from matplotlib.pyplot import subplots, show
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from ipywidgets import interact, fixed
from .lib.signal_plot import hist_plot
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


def sampling_demo1_plot(axes, distX=distributions[0], muX=0, sigmaX=1, N=1000):

    Nx = 801
    x = np.linspace(-10, 10, Nx)

    fX = pdf(x, muX, sigmaX, distX)
    FX = cumulative_trapezoid(fX, x, initial=0)

    interp = interp1d(FX, x, kind='linear', bounds_error=False,
                      fill_value=x[-1])

    samples = interp(np.random.rand(N))

    axes.clear()
    hist_plot(x, samples, density=True, axes=axes)
    axes.plot(x, fX, label='desired')
    axes.set_xlim(-5, 5)


def sampling_demo1():

    fig, axes = subplots(1)
    show()

    interact(sampling_demo1_plot, axes=fixed(axes), distX=distributions,
             muX=(-2, 2), sigmaX=(0.01, 5, 0.01),
             N=[100, 1000, 10000, 100000, 1000000],  continuous_update=False)
