import numpy as np
from scipy.optimize import minimize
from EnerTools import *
import matplotlib.pyplot as plt

def quasi_newton(f,x0,grad_func):
    resultat=minimize(f,x0,method="BFGS",jac=lambda x : grad_func(f,x),options={"return_all":True})
    return resultat

def infos_opti(res):
    if res.success:
        return res.x,res.fun,res.nit
    return "Optimisation failed"

def visualisation_nwtn(res,f):
    energies = []

    for x in res.allvecs:
        energies.append(f(x))

    iterations = np.arange(len(energies))
    fig, ax = plt.subplots()
    ax.set_title("Evolution de l'énergie par itération")
    ax.plot(iterations, energies)
    ax.set_xlabel("Itération")
    ax.set_ylabel("Énergie")
    plt.show()