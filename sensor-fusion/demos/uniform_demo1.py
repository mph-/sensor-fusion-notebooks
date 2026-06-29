# M. P. Hayes UCECE
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from .lib.signal_plot import signal_plot


def uniform_demo1_plot(axes, a=-1, b=1, autoscale=False):

    N = 401
    x = np.linspace(-10, 10, N)

    fX = 1.0 * ((x >= a) & (x <= b)) / (abs(b - a) + 1e-12)

    ylim = None
    if not autoscale:
        ylim = [0, 0.55]

    axes.clear()
    signal_plot(x, fX, ylim=ylim, axes=axes)
    mu_X = 0.5 * (a + b)
    sigma_X = abs(b - a) / np.sqrt(12)
    if b < a:
        mu_X = 0
        sigma_X = 0
    axes.set_title(r'$\mu_{X} = %.1f, \sigma_{X} = %.1f$' %
                   (mu_X, sigma_X))


def uniform_demo1():

    fig, axes = subplots(1)
    show()

    interact(uniform_demo1_plot, axes=fixed(axes), a=(-5, 5), b=(-5, 5))
