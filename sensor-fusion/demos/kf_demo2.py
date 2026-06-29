# Michael P. Hayes UCECE, Copyright 2018--2022
import numpy as np
from ipywidgets import interact
from matplotlib.pyplot import subplots, show as show1
from .lib.utils import gauss

steps = 10
show_choices = ['Estimates', 'PDFs', 'Estimates and PDFs']


def kf_demo2_plot(show=show_choices[-1], v=2.0, sigmaX0=0.1, sigmaV=1,
                  sigmaW=0.4, seed=1, step=1):

    np.random.seed(seed)

    dt = 1

    A = 1
    B = dt
    C = 1
    D = 0

    Nx = 801
    x = np.linspace(-10, 40, Nx)

    xrobot = 0
    Xpriormean = 0
    Xinitialmean = 0
    Xinitialvar = sigmaX0**2

    if show == show_choices[2]:
        fig, axes = subplots(2, figsize=(10, 6))
        pdfax, estax = axes
    elif show == show_choices[1]:
        fig, pdfax = subplots(1, figsize=(10, 6))
        estax = None
    else:
        fig, estax = subplots(1, figsize=(10, 6))
        pdfax = None

    if estax is not None:
        estax.set_xlim(0, steps)
        estax.set_ylim(0, steps * B * v)
        estax.grid(True)
        estax.plot(np.nan, np.nan, '-o', color='C0', label='actual')
        estax.plot(np.nan, np.nan, 'x', color='C1', label='prior')
        estax.plot(np.nan, np.nan, '+', color='C2', label='likelihood')
        estax.plot(0, Xinitialmean, '*', color='C3', label='posterior')
        estax.set_ylabel('Estimate')
        estax.legend()

    mx = (x < round(steps * B * v)) & (x > -2)

    Xpostmean = 0
    Xpostvar = 0

    for m in range(1, step + 1):

        # Note, making the robot have some process noise
        # will also affect the measurement.
        xrobot += B * v + np.random.randn(1) * sigmaW

        Xpriormean += B * v

        if m > 1:
            Xinitialmean = Xpostmean
            Xinitialvar = Xpostvar

        Xpriormean = A * Xinitialmean + B * v
        Xpriorvar = (A**2) * Xinitialvar + sigmaW**2

        z = C * xrobot + np.random.randn(1) * sigmaV

        Xinfermean = (z - D * v) / C
        Xinfervar = (sigmaV**2) / (C**2)

        K = Xpriorvar / (Xpriorvar + Xinfervar)

        Xpostmean = K * Xinfermean + (1 - K) * Xpriormean
        Xpostvar = (Xpriorvar * Xinfervar) / (Xpriorvar + Xinfervar)

        if estax is not None:
            estax.plot(m, xrobot, 'o', color='C0')
            estax.plot(m, Xpriormean, 'x', color='C1')
            estax.plot(m, Xinfermean, '+', color='C2')
            estax.plot(m, Xpostmean, '*', color='C3')

    fXinitial = gauss(x, Xinitialmean, np.sqrt(Xinitialvar))
    fXprior = gauss(x, Xpriormean, np.sqrt(Xpriorvar))
    fXinfer = gauss(x, Xinfermean, np.sqrt(Xinfervar))
    fXpost = gauss(x, Xpostmean, np.sqrt(Xpostvar))

    if pdfax is not None:
        pdfax.plot(x[mx], fXinitial[mx], ':',
                   label='$f_{X_{%d}}$ initial' % (m - 1))
        pdfax.plot(x[mx], fXprior[mx], '--',
                   label='$f_{X_{%d}^{-}}$ prior' % m)
        pdfax.plot(x[mx], fXinfer[mx], '-.',
                   label='$L_{%d}$ likelihood' % m)
        pdfax.plot(x[mx], fXpost[mx], label='$f_{X_{%d}^{+}}$ posterior' % m)
        pdfax.grid(True)
        pdfax.legend()

    if pdfax is not None:
        pdfax.set_title('z=%.2f, K=%.2f' % (z, K))
    else:
        estax.set_title('z=%.2f, K=%.2f' % (z, K))

    show1()


def kf_demo2():

    interact(kf_demo2_plot, show=show_choices,
             step=(1, steps), v=(1.0, 4.0, 0.2),
             sigmaX0=(0.1, 2, 0.1),
             sigmaV=(0.1, 2, 0.1),
             sigmaW=(0.1, 2, 0.1),
             seed=(1, 100, 1),
             continuous_update=False)
