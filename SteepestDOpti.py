import numpy as np
from scipy.optimize import line_search
from EnerTools import gradient_calc,direction,pas_optimal
import CalculeEnergie
import matplotlib.pyplot as plt

def SteepestDescent(f,x0,itermax=1000,cutof=1e-5):
    compt=0
    epslion=np.inf
    total_cords=[]
    total_norms=[]
    while compt<itermax and epslion>cutof:
        grd=gradient_calc(f,x0)
        way=direction(grd)
        epslion=np.linalg.norm(grd)

        total_cords.append(x0)
        total_norms.append(epslion)
        if epslion<cutof:
            print ("Convergence")
            break

        grad_func = lambda x: gradient_calc(f, x)
        step=pas_optimal(f,grad_func,x0,way)
        if step is None:
            print("No step found")
            break

        epslion=step
        total_cords.append(x0)
        next=x0.copy()+step*way

        x0=next
        compt+=1

    return total_cords,total_norms

def visualisation(coords):
    iter=np.arange(len(coords))
    fig,ax=plt.subplots()
    ax.set_title("Evolution de la norme par iteration")
    ax.plot(iter,coords,color='red')
    ax.set_xlabel("iteration",rotation=90)
    ax.set_ylabel("norme a al coordonee")
    plt.show()
