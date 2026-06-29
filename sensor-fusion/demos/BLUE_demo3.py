# M. P. Hayes UCECE
import numpy as np
from matplotlib.pyplot import subplots, show
from ipywidgets import interact, fixed
from .lib.signal_plot import signal_overplot3
from .lib.utils import gauss


def BLUE_demo3_plot(axes, muX1=2, muX2=2, sigmaX1=1.0, sigmaX2=2.0, w1=0.5):

    w2 = 1.0 - w1

    muX = w1 * muX1 + w2 * muX2
    sigmaX = np.sqrt(w1**2 * sigmaX1**2 + w2**2 * sigmaX2**2)

    N = 401
    x = np.linspace(-8, 8, N)

    fX1 = gauss(x, muX1, sigmaX1)
    fX2 = gauss(x, muX2, sigmaX2)
    fX = gauss(x, muX, sigmaX)

    axes.clear()
    fig = signal_overplot3(x, fX1, x, fX2, x, fX,
                           (r'$f_{\hat{X}_1}(\hat{x})$',
                            r'$f_{\hat{X}_2}(\hat{x})$',
                            r'$f_{\hat{X}}(\hat{x})$'), ylim=(0, 0.5),
                           axes=axes)
    fig.axes[0].set_xlabel(r'$\hat{x}$')
    fig.axes[0].grid(True)


def BLUE_demo3():

    fig, axes = subplots(1)
    show()

    interact(BLUE_demo3_plot, axes=fixed(axes),
             sigmaX1=(0.5, 4.0, 0.1),
             sigmaX2=(0.5, 4.0, 0.1),
             w1=(0, 1.0, 0.05))
