# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from matplotlib.pyplot import subplots, show
from scipy.interpolate import interp1d
from ipywidgets import interact, fixed
from .lib.signal_plot import hist_plot2
from .lib.utils import gauss
from .lib.kde import KDE

distributions = ['gaussian', 'uniform']


def pdf(x, muX, sigmaX, distribution):

    if distribution == 'gaussian':
        return gauss(x, muX, sigmaX)
    elif distribution == 'uniform':
        xmin = muX - np.sqrt(12) * sigmaX / 2
        xmax = muX + np.sqrt(12) * sigmaX / 2
        return 1.0 * ((x >= xmin) & (x <= xmax)) / (xmax - xmin)
    raise ValueError('Unknown distribution %s' % distribution)


def sampling_demo3_plot(axes, distX=distributions[0], muX=0, sigmaX=1,
                        N=1000, seed=1):

    np.random.seed(seed)

    Nx = 801
    x = np.linspace(-10, 10, Nx)
    dx = x[1] - x[0]

    fX = pdf(x, muX, sigmaX, distX)
    FX = np.cumsum(fX) * dx

    interp = interp1d(FX, x, kind='linear', bounds_error=False,
                      fill_value=x[-1])

    xsamples = interp(np.random.rand(N))
    zsamples = xsamples**2

    fXest = KDE(xsamples).estimate(x)
    fZest = KDE(zsamples).estimate(x)

    for ax in axes:
        ax.clear()

    hist_plot2(x, xsamples, x, zsamples, density=True, axes=axes)
    axes[0].plot(x, fX, label='desired')
    axes[0].plot(x, fXest, label='estimated')
    axes[0].set_xlim(-5, 5)
    axes[0].legend()

    axes[1].plot(x, fZest, label='estimated')
    axes[1].set_xlim(-5, 5)
    axes[1].legend()


def sampling_demo3():

    fig, axes = subplots(2)
    show()

    interact(sampling_demo3_plot, axes=fixed(axes), distX=distributions,
             muX=(-2, 2), sigmaX=(0.01, 5, 0.01),
             N=[10, 100, 1000, 10000, 100000, 1000000],
             seed=(1, 100, 1),
             continuous_update=False)
