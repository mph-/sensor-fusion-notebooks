# M. P. Hayes UCECE
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from .lib.utils import mgauss2


def bivariate_gaussian_demo2_plot(axes, muX=0, sigmaX=1, muY=0, sigmaY=1,
                                  rhoXY=0):

    N = 101
    x = np.linspace(-10, 10, N)
    y = np.linspace(-10, 10, N)

    if sigmaX < 0.01:
        sigmaX = 0.001
    if sigmaY < 0.01:
        sigmaY = 0.001

    X, Y = np.meshgrid(x, y)

    fXY = mgauss2(x, y, (muX, muY), (sigmaX, sigmaY), rhoXY)

    axes.clear()
    axes.contour(X, Y, fXY)
    axes.axis('equal')

    axes.set_xlabel('$X$')
    axes.set_ylabel('$Y$')


def bivariate_gaussian_demo2():

    fig, axes = subplots(1)
    show()

    interact(bivariate_gaussian_demo2_plot, axes=fixed(axes),
             muX=(-5, 5), sigmaX=(0, 5, 0.1),
             muY=(-5, 5), sigmaY=(0, 5, 0.1), rhoXY=(-0.95, 0.99, 0.05))
