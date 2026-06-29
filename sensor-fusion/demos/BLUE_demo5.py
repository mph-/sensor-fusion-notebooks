# M. P. Hayes UCECE
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show


def BLUE_demo5_plot(axes, sigmaX1=1.0, sigmaX2=2.0):

    w1 = np.linspace(0, 1, 201)
    w2 = 1 - w1

    sigmaX = np.sqrt(w1**2 * sigmaX1**2 + w2**2 * sigmaX2**2)

    axes.clear()
    axes.plot(w1, sigmaX)
    axes.set_xlabel('Weight $w_1$')
    axes.set_ylabel(r'Std dev. $\sigma_{\hat{X}}$')
    axes.grid(True)


def BLUE_demo5():

    fig, axes = subplots(1)
    show()

    interact(BLUE_demo5_plot, axes=fixed(axes),
             sigmaX1=(0.1, 4.0, 0.1),
             sigmaX2=(0.1, 4.0, 0.1),
             continuous_update=False)
