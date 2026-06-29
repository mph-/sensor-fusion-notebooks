# M. P. Hayes UCECE
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from .lib.utils import mgauss2


def bivariate_gaussian_demo3_plot(axes, muX=0, sigmaX=1, muY=0, sigmaY=1,
                                  rhoXY=0, slice='x = 0'):

    N = 101
    x = np.linspace(-10, 10, N)
    y = np.linspace(-10, 10, N)

    X, Y = np.meshgrid(x, y)

    fXY = mgauss2(x, y, (muX, muY), (sigmaX, sigmaY), rhoXY)

    axes.clear()
    if slice == 'x = 0':
        axes.plot(x, fXY[N // 2, :])
        axes.set_xlabel('$X$')
    elif slice == 'y = 0':
        axes.plot(y, fXY[:, N // 2])
        axes.set_xlabel('$Y$')
    else:
        raise ValueError('Unknown slice ' + slice)


def bivariate_gaussian_demo3():

    fig, axes = subplots(1, figsize=(4, 4))
    show()

    interact(bivariate_gaussian_demo3_plot, axes=fixed(axes),
             muX=(-5, 5), sigmaX=(0.01, 5, 0.01),
             muY=(-5, 5), sigmaY=(0.01, 5, 0.01), rhoXY=(-0.95, 0.99, 0.05),
             slice=['x = 0', 'y = 0'])
