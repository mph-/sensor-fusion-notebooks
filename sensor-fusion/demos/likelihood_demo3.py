# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact
from matplotlib.pyplot import subplots, show
from .lib.utils import gauss


def likelihood_demo3_plot(sigmaV=0.2, z=2, show_Lambda=True):

    Nx = 801
    x = np.linspace(0, 5, Nx)

    d = 0.5
    a = 5.0 / d

    h = (a * x) * (x <= d) + (a * d**2 / (x + 1e-6)) * (x > d)

    fZgX = gauss(z - h, 0, sigmaV)

    m1 = x < d
    m2 = x >= d

    x1 = np.interp(z, h[m1], x[m1])
    # Can only interpolate a monotonic function, so
    # invert values to achieve a monotonic function.
    x2 = np.interp(1 / z, 1 / h[m2], x[m2])

    # FIXME Dynamically change number of axes.  This creates
    # a new figure for every call
    fig, axes = subplots(2 + show_Lambda * 1, figsize=(10, 5))
    fig.tight_layout()
    show()

    axes[0].plot(x, h, color='orange', label='$h(x)$')
    axes[0].plot(x1, z, 'o', color='orange')
    axes[0].plot(x2, z, 'o', color='orange')
    axes[0].grid(True)
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$z$')
    axes[0].legend()

    axes[1].grid(True)
    axes[1].plot(x, fZgX, '--', label='$L(x|%.1f)$' % z)
    axes[1].grid(True)
    axes[1].set_xlabel('$x$')
    axes[1].legend()

    if show_Lambda:
        zv = np.linspace(0, 5, 201)

        X, Z = np.meshgrid(x, zv)
        H = (a * X) * (X <= d) + (a * d**2 / (X + 1e-6)) * (X > d)
        Lambda = gauss(Z - H, 0, sigmaV)

        axes[2].imshow(Lambda, origin='lower',
                       extent=(x[0], x[-1], zv[0], zv[-1]))
        axes[2].axis('tight')
        axes[2].set_xlabel('$x$')
        axes[2].set_ylabel('$z$')


def likelihood_demo3():
    interact(likelihood_demo3_plot, sigmaV=(0.01, 0.5, 0.01), z=(0, 5, 0.2))
