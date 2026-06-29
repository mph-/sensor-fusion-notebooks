# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from .lib.signal_plot import signal_plot


def linearise_demo1_plot(axes, x0=1):

    Nx = 201
    x = np.linspace(0, 10, Nx)

    k1 = 10
    k2 = 1

    h = k1 / (k2 + x)

    J = -k1 / (k2 + x0)**2

    h2 = J * (x - x0) + k1 / (k2 + x0)

    axes.clear()
    signal_plot(x, h, axes=axes)
    axes.set_xlabel('$x$')
    axes.plot(x, h2)
    axes.set_ylim(0, k1)


def linearise_demo1():

    fig, axes = subplots(1)
    show()

    interact(linearise_demo1_plot, axes=fixed(axes), x0=(0, 10))
