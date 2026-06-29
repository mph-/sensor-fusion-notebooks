# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from scipy.interpolate import interp1d
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from IPython.display import display
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


def sampling_demo0_plot(axes, distX=distributions[0], muX=0, sigmaX=1,
                        seed=1, N=2, show_histogram=False):

    Nx = 801
    x = np.linspace(-10, 10, Nx)
    dx = x[1] - x[0]

    fX = pdf(x, muX, sigmaX, distX)
    FX = np.cumsum(fX) * dx

    np.random.seed(seed)

    interp = interp1d(FX, x, kind='linear', bounds_error=False,
                      fill_value=x[-1])

    samples = interp(np.random.rand(N))
    values = ', '.join(['%.1f' % sample for sample in samples])

    display(values)
    display('est muX = %.1f  est sigmaX = %.1f' % (np.mean(samples),
                                                   np.std(samples)))

    axes.clear()
    hist_plot(x, samples, density=True, axes=axes)
    axes.plot(x, fX, label='desired')
    axes.set_xlim(-5, 5)


def sampling_demo0():

    fig, axes = subplots(1)
    show()

    interact(sampling_demo0_plot, axes=fixed(axes), distX=distributions,
             muX=(-2, 2), sigmaX=(0.01, 5, 0.01), seed=(1, 100, 1),
             N=(2, 100, 2),  continuous_update=False)
