import numpy as np
from scipy.optimize import line_search

def gradient_calc(f,x,h=1e-5):
    grd=np.zeros(x.shape)
    for i in range(x.shape[0]):
        xp=np.copy(x)
        xp[i]+=h
        grd[i]=(f(xp)-f(x))/h

    return grd

def direction(grad):
    inv=np.full(grad.shape,-1)
    return grad*inv

def pas_optimal(f,gradient_calc,x,direction):
    return line_search(f,gradient_calc,x,direction)[0]

