# M. P. Hayes UCECE
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from .lib.signal_plot import signal_plot
from .lib.utils import gauss


def gauss_demo1_plot(axes, muX=0, sigmaX=1, autoscale=False):

    N = 401
    x = np.linspace(-10, 10, N)

    fX = gauss(x, muX, sigmaX)

    ylim = None
    if not autoscale:
        ylim = [0, 0.55]

    axes.clear()
    signal_plot(x, fX, ylim=ylim, axes=axes)


def gauss_demo1():

    fig, axes = subplots(1)
    show()

    interact(gauss_demo1_plot, axes=fixed(axes), muX=(-5, 5),
             sigmaX=(0.01, 5, 0.01))
