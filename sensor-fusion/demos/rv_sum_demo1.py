# M. P. Hayes UCECE
import numpy as np
from matplotlib.pyplot import subplots, show
from ipywidgets import interact, fixed
from .lib.signal_plot import signal_plot
from .lib.utils import gauss, pdf, distributions


def rv_sum_demo1_plot(axes, muX=0, sigmaX=1, N=5,
                      distribution=distributions[1]):

    # This is subject to numerical error especially with a uniform
    # distribution.

    Nx = 501
    x = np.linspace(-10, 10, Nx)
    dx = x[1] - x[0]

    fX = pdf(x, muX, sigmaX, distribution)
    fZ = fX
    for n in range(1, N):
        fZ = np.convolve(fZ, fX) * dx

    M = fZ.shape[-1]
    x = np.arange(-M // 2, M // 2) * dx
    mx = (x < 8) & (x > -8)

    fG = gauss(x, muX * N, sigmaX * np.sqrt(N))

    axes.clear()
    signal_plot(x[mx], fZ[mx], axes=axes)
    axes.plot(x[mx], fG[mx], '--')


def rv_sum_demo1():

    fig, axes = subplots(1)
    show()

    interact(rv_sum_demo1_plot, axes=fixed(axes),
             muX=(-2, 2), sigmaX=(0.01, 2, 0.01),
             distribution=distributions, N=(1, 100))
