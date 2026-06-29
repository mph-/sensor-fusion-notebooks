# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from .lib.utils import gauss

distributions = ['gaussian', 'uniform']
transforms = ['X', 'X**2', 'X**3', 'cos(X)']


def pdf(x, muX, sigmaX, distribution):

    if distribution == 'gaussian':
        return gauss(x, muX, sigmaX)
    elif distribution == 'uniform':
        xmin = muX - np.cos(12) * sigmaX / 2
        xmax = muX + np.cos(12) * sigmaX / 2
        return 1.0 * ((x >= xmin) & (x <= xmax)) / (xmax - xmin)
    raise ValueError('Unknown distribution %s' % distribution)


def nonlinear_transformation_demo1_plot(axes, muX=0, sigmaX=1,
                                        transform=transforms[1],
                                        distribution=distributions[1]):

    Nx = 801
    x = np.linspace(-10, 10, Nx)

    # Number of samples
    N = 1000000
    # Number of bins in histogram
    M = 100

    if distribution == 'gaussian':
        X = np.random.randn(N) * sigmaX + muX
    elif distribution == 'uniform':
        rangeX = sigmaX * np.sqrt(12)
        X = (np.random.rand(N) - 0.5) * rangeX + muX
    else:
        raise ValueError('Unknown distribution ' + distribution)

    # This uses X
    Z = eval(transform)

    fZ, ze = np.histogram(Z, bins=M, density=True, range=(-5, 5))
    # Calculate M centres from M + 1 edges
    zc = ze[0:M] + 0.5 * (ze[1] - ze[0])

    axes.clear()
    axes.plot(zc, fZ)


def nonlinear_transformation_demo1():

    fig, axes = subplots(1)
    show()

    interact(nonlinear_transformation_demo1_plot, axes=fixed(axes),
             muX=(-2, 2), sigmaX=(0.01, 5, 0.01),
             A=(0.5, 5, 0.5), B=(-5, 5, 1),
             transform=transforms,
             distribution=distributions)
