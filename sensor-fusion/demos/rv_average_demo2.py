# M. P. Hayes UCECE
import numpy as np
import numpy.ma as ma
from matplotlib.pyplot import show
from ipywidgets import interact
from .lib.signal_plot import signal_plot
from .lib.utils import gauss

distributions = ['gaussian', 'uniform']


def pdf(x, muX, sigmaX, distribution):

    if distribution == 'gaussian':
        return gauss(x, muX, sigmaX)
    elif distribution == 'uniform':
        xmin = muX - np.sqrt(12) * sigmaX / 2
        xmax = muX + np.sqrt(12) * sigmaX / 2
        u = 1.0 * ((x >= xmin) & (x <= xmax))
        u /= np.trapz(u, x)
        return u
    raise ValueError('Unknown distribution %s' % distribution)


def rv_average_demo2_plot(muX=0, sigmaX=1, N=5, distribution=distributions[1]):

    Nx = 201
    x = np.linspace(-5, 5, Nx)
    dx = x[1] - x[0]

    fX = pdf(x, muX, sigmaX, distribution)
    fZ = fX
    for n in range(1, N):
        fZ = np.convolve(fZ, fX) * dx

    fZ = fZ[::N] * N

    fG = gauss(x, muX / N, sigmaX / np.sqrt(N))

    mx = (x < 5) & (x > -5)

    lfZ = ma.log(fZ)
    lfG = ma.log(fG)

    fig = signal_plot(x[mx], lfZ[mx])
    fig.axes[0].plot(x[mx], lfG[mx], '--')
    fig.axes[0].set_ylim(-35, 5)

    show()


def rv_average_demo2():
    interact(rv_average_demo2_plot, muX=(-2, 2), sigmaX=(0.01, 5, 0.01),
             distribution=distributions, N=(1, 100))
