# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed, interactive, fixed
from matplotlib.pyplot import figure
from .lib.utils import gauss

def kf_demo1_plot(step=1, v=2.0, sigmaX0=0.1, sigmaV=0.4, sigmaW=0.4,
                  seed=1):

    np.random.seed(seed)

    dt = 1

    A = 1
    B = dt
    C = 1
    D = 0

    Nx = 801
    x = np.linspace(-10, 40, Nx)

    fig = figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.grid(True)

    mx = (x < 12) & (x > -2)

    Xinitialmean = 0
    Xinitialvar = sigmaX0**2

    for m in range(1, step + 1):

        if m > 1:
            Xinitialmean = Xpostmean
            Xinitialvar = Xpostvar

        Xpriormean = A * Xinitialmean + B * v
        Xpriorvar = (A**2) * Xinitialvar + sigmaW**2

        # Hack
        z = C * Xpriormean + np.random.randn(1) * sigmaV

        Xinfermean = (z - D * v) / C
        Xinfervar = (sigmaV**2) / (C**2)

        K = Xpriorvar / (Xpriorvar + Xinfervar)

        Xpostmean = K * Xinfermean + (1 - K) * Xpriormean
        Xpostvar = (Xpriorvar * Xinfervar) / (Xpriorvar + Xinfervar)

    fXinitial = gauss(x, Xinitialmean, np.sqrt(Xinitialvar))
    fXprior = gauss(x, Xpriormean, np.sqrt(Xpriorvar))
    fXinfer = gauss(x, Xinfermean, np.sqrt(Xinfervar))
    fXpost = gauss(x, Xpostmean, np.sqrt(Xpostvar))

    ax.plot(x[mx], fXinitial[mx], ':', label='$f_{X_{%d}}$ initial' % (m - 1))
    ax.plot(x[mx], fXprior[mx], '--', label='$f_{X_{%d}^{-}}$ prior' % m)
    ax.plot(x[mx], fXinfer[mx], '-.', label='$L_{%d}$ likelihood' % m)
    ax.plot(x[mx], fXpost[mx], label='$f_{X_{%d}^{+}}$ posterior' % m)

    ax.legend()
    ax.text(z, max(fXpost) + 0.5, 'z=%.2f' % z)

def kf_demo1():
    interact(kf_demo1_plot, step=(1, 5), v=(1.0, 4.0, 0.2),
             sigmaX0=(0.1, 1, 0.1),
             sigmaV=(0.1, 1, 0.1),
             sigmaW=(0.1, 1, 0.1),
             seed=(1, 100, 1),
             continuous_update=False)
