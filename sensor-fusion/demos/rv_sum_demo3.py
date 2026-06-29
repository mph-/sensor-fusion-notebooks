# M. P. Hayes UCECE
import numpy as np
from ipywidgets import interact, fixed
from .lib.signal_plot import signal_plot3
from .lib.utils import gauss, pdf, distributions
from matplotlib.pyplot import subplots, show


def rv_sum_demo3_plot(axes, muX=0, rangeX=2, muY=0, rangeY=2):

    N = 401
    x = np.linspace(-5, 5, N)
    dx = x[1] - x[0]

    xmin = muX - 0.5 * rangeX
    xmax = muX + 0.5 * rangeX

    ymin = muY - 0.5 * rangeY
    ymax = muY + 0.5 * rangeY

    fX = 1.0 * ((x >= xmin) & (x <= xmax))
    fX /= np.trapz(fX, x)

    fY = 1.0 * ((x >= ymin) & (x <= ymax))
    fY /= np.trapz(fY, x)

    fZ = np.convolve(fX, fY) * dx

    M = fZ.shape[-1]
    z = np.arange(-M // 2, M // 2) * dx
    mz = (z < 5) & (z > -5)

    mx = (x < 5) & (x > -5)

    for ax in axes:
        ax.clear()

    signal_plot3(x[mx], fX[mx], x[mx], fY[mx],
                 z[mz], fZ[mz], ylim=(0, 0.55), axes=axes)
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    axes[2].set_xlabel('x')

    axes[0].set_ylabel('$f_X(x)$')
    axes[1].set_ylabel('$f_Y(x)$')
    axes[2].set_ylabel('$f_{X+Y}(x)$')


def rv_sum_demo3():

    fig, axes = subplots(3)
    show()

    interact(rv_sum_demo3_plot, axes=fixed(axes),
             muX=(-2, 2), rangeX=(1, 4, 0.5),
             muY=(-2, 2), rangeY=(1, 4, 0.5),
             continuous_update=False)
