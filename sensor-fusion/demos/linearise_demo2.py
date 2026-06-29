# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show


def linearise_demo2_plot(axes, x0=1):

    Nx = 201
    x = np.linspace(0, 10, Nx)

    a = 1
    b = -9
    c = 27
    d = 100

    h = a * x**3 + b * x**2 + c * x + d

    J = 3 * a * x0**2 + 2 * b * x0 + c
    u = a * x0**3 + b * x0**2 + c * x0 + d
    h2 = (x - x0) * J + u

    axes.clear()
    axes.set_xlabel('$x$')
    axes.plot(x, h2)
    axes.set_ylim(0, max(max(h), max(h2)))


def linearise_demo2():

    fig, axes = subplots(1)
    show()

    interact(linearise_demo2_plot, axes=fixed(axes), x0=(0, 10))
