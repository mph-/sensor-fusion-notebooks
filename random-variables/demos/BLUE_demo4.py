# M. P. Hayes UCECE
import numpy as np
from ipywidgets import interact, interactive, fixed
from .lib.signal_plot import signal_plot3
from .lib.utils import gauss, rect, trap2

def BLUE_demo4_plot(sigmaX1=1.0, sigmaX2=2.0, w1=0.5):

    w2 = 1.0 - w1
    
    muX1 = 2
    muX2 = 2

    WX1 = sigmaX1 * np.sqrt(12)
    WX2 = sigmaX2 * np.sqrt(12)

    muX = w1 * muX1 + w2 * muX2
    sigmaX = np.sqrt(w1**2 * sigmaX1**2 + w2**2 * sigmaX2**2)    
    WX = w1 * WX1 + w2 * WX2
    TX = abs(w1 * WX1 - w2 * WX2)
    
    N = 401
    x = np.linspace(-8, 8, N)
    
    fX1 = rect((x - muX1) / WX1) / WX1
    fX2 = rect((x - muX2) / WX2) / WX2
    fX = trap2(x, TX, WX) * (2 / (TX + WX))
    
    signal_plot3(x, fX1, x, fX2, x, fX)

def BLUE_demo4():
    interact(BLUE_demo4_plot,
             sigmaX1=(0.1, 4.0, 0.1),
             sigmaX2=(0.1, 4.0, 0.1),
             w1=(0, 1.0, 0.1))
    
    

    

