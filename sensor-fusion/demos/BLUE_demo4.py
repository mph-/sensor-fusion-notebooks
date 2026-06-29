# M. P. Hayes UCECE
import numpy as np
from matplotlib.pyplot import subplots, show
from ipywidgets import interact, fixed
from .lib.signal_plot import signal_overplot3
from .lib.utils import rect, trap2


def BLUE_demo4_plot(axes, muX1=2, muX2=2, sigmaX1=1.0, sigmaX2=2.0, w1=0.5):

    w2 = 1.0 - w1

    WX1 = sigmaX1 * np.sqrt(12)
    WX2 = sigmaX2 * np.sqrt(12)

    muX = w1 * muX1 + w2 * muX2
    WX = w1 * WX1 + w2 * WX2
    TX = abs(w1 * WX1 - w2 * WX2)

    N = 401
    x = np.linspace(-8, 8, N)

    fX1 = rect((x - muX1) / WX1) / WX1
    fX2 = rect((x - muX2) / WX2) / WX2
    fX = trap2(x - muX, TX, WX) * (2 / (TX + WX))

    axes.clear()
    signal_overplot3(x, fX1, x, fX2, x, fX,
                     (r'$f_{\hat{X}_1}(\hat{x})$',
                      r'$f_{\hat{X}_2}(\hat{x})$',
                      r'$f_{\hat{X}}(\hat{x})$'), ylim=(0, 0.5),
                     axes=axes)
    axes.set_xlabel(r'$\hat{x}$')
    axes.grid(True)


def BLUE_demo4():

    fig, axes = subplots(1)
    show()

    interact(BLUE_demo4_plot, axes=fixed(axes),
             sigmaX1=(0.5, 4.0, 0.1),
             sigmaX2=(0.5, 4.0, 0.1),
             w1=(0, 1.0, 0.05))
