# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from matplotlib.pyplot import show
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from ipywidgets import interact, interactive, fixed, interact
from .lib.signal_plot import hist_plot
from .lib.utils import rect, sinc, gauss

distributions = ['gaussian', 'uniform']


def pdf(x, muX, sigmaX, distribution):

    if distribution == 'gaussian':
        return gauss(x, muX, sigmaX)
    elif distribution == 'uniform':
        xmin = muX - np.sqrt(12) * sigmaX / 2
        xmax = muX + np.sqrt(12) * sigmaX / 2
        return 1.0 * ((x >= xmin) & (x <= xmax)) / (xmax - xmin)
    raise ValueError('Unknown distribution %s' % distribution)

def sampling_demo1_plot(distX=distributions[0], muX=0, sigmaX=1, N=1000):

    Nx = 801
    x = np.linspace(-10, 10, Nx)
    dx = x[1] - x[0]

    fX = pdf(x, muX, sigmaX, distX)
    FX = cumulative_trapezoid(fX, x, initial=0)

    interp = interp1d(FX, x, kind='linear', bounds_error=False,
                      fill_value=x[-1])

    samples = interp(np.random.rand(N))

    fig = hist_plot(x, samples, density=True)
    axes = fig.axes
    axes[0].plot(x, fX, label='desired')
    axes[0].set_xlim(-5, 5)

def sampling_demo1():
    interact(sampling_demo1_plot, distX=distributions,
             muX=(-2, 2), sigmaX=(0.01, 5, 0.01),
             N=[100, 1000, 10000, 100000, 1000000],  continuous_update=False)
