# Michael P. Hayes UCECE, Copyright 2018--2023
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from .lib.utils import gauss


def hf_demo1_plot(axes, sigmaX0=0.4, sigmaV=0.1, sigmaW=0.1,
                  seed=1, M=50):

    np.random.seed(seed)

    v = 2
    dt = 1

    def h(x):
        """Sensor model"""
        return x

    def g(x, v):
        """Motion model"""
        return x + v * dt

    # Tweak number of bins so that user has expected number of
    # bins spanning the plotted range for x.
    M = int(M * (4 - -2) / (3 - -1))

    x = np.linspace(-2, 4, M)
    dx = x[1] - x[0]

    fX_initial = gauss(x, 0, sigmaX0)
    fX_prior = np.zeros(M)

    # Simulate measurement
    z = v * dt + np.random.randn(1) * sigmaV

    # Predict step: "Convolve" initial belief histogram with conditional
    # PDF for process model
    for i in range(len(x)):
        total = 0
        for j in range(len(x)):
            total += fX_initial[j] * \
                gauss(x[i] - g(x[j], v), 0, sigmaW)
        fX_prior[i] = total * dx

    # Update step: Apply Bayes' theorem
    fX_posterior = fX_prior * gauss(z - h(x), 0, sigmaV)
    eta = 1 / np.trapz(fX_posterior, x)
    fX_posterior *= eta

    axes.clear()
    axes.grid(True)

    axes.bar(x, fX_initial, label='$X_0$ initial',
             edgecolor='black', width=dx)

    axes.bar(x, fX_prior, label='$X_1^{-}$ prior',
             edgecolor='black', width=dx)

    axes.bar(x, fX_posterior,
             label='$X_1^{+}$ posterior', edgecolor='black', width=dx)

    axes.set_xlim(-1, 3)
    axes.set_ylim(0, 4)
    axes.set_xlabel('Position')
    axes.set_ylabel('Prob. density')
    axes.legend()


def hf_demo1():

    fig, axes = subplots(1, figsize=(10, 5))
    show()

    interact(hf_demo1_plot, axes=fixed(axes), step=(1, 5), M=(10, 200, 10),
             v=(1.0, 4.0, 0.2),
             sigmaX0=(0.1, 1, 0.1),
             sigmaV=(0.1, 1, 0.1),
             sigmaW=(0.1, 1, 0.1),
             seed=(1, 100, 1))
