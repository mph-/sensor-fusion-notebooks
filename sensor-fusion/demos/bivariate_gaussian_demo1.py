# M. P. Hayes UCECE
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from matplotlib import cm
from .lib.utils import mgauss2


def bivariate_gaussian_demo1_plot(axes, muX=0, sigmaX=1, muY=0, sigmaY=1,
                                  rhoXY=0):

    N = 101
    x = np.linspace(-10, 10, N)
    y = np.linspace(-10, 10, N)

    X, Y = np.meshgrid(x, y)

    fXY = mgauss2(x, y, (muX, muY), (sigmaX, sigmaY), rhoXY)

    axes.clear()
    axes.plot_surface(X, Y, fXY, rstride=1, cstride=1, cmap=cm.jet,
                      linewidth=0, antialiased=False)

    axes.set_xlabel('$X$')
    axes.set_ylabel('$Y$')


def bivariate_gaussian_demo1():

    fig, axes = subplots(1, subplot_kw=dict(projection='3d'))
    show()

    interact(bivariate_gaussian_demo1_plot, axes=fixed(axes), muX=(-5, 5),
             sigmaX=(0.01, 5, 0.01), muY=(-5, 5), sigmaY=(0.01, 5, 0.01),
             rhoXY=(-0.95, 0.99, 0.05))
