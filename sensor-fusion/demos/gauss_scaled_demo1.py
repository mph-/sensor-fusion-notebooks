# M. P. Hayes UCECE
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from .lib.signal_plot import signal_plot2
from .lib.utils import gauss


def gauss_scaled_demo1_plot(axes, muX=0, sigmaX=1, a=1, autoscale=False):

    if a == 0:
        a = 1e-3

    N = 401
    x = np.linspace(-10, 10, N)

    muY = a * muX
    sigmaY = abs(a) * sigmaX

    fX = gauss(x, muX, sigmaX)
    fY = gauss(x, muY, sigmaY)

    ylim = None
    if not autoscale:
        ylim = [0, 0.55]

    axes[0].clear()
    axes[1].clear()
    signal_plot2(x, fX, x, fY, ylim=ylim, axes=axes)


def gauss_scaled_demo1():

    fig, axes = subplots(2)
    fig.tight_layout()
    show()

    interact(gauss_scaled_demo1_plot, axes=fixed(axes),
             muX=(-5, 5), sigmaX=(0.01, 5, 0.01),
             a=(0.0, 5, 0.1))
