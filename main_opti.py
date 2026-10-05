import numpy as np
import time
from CalculeEnergie import *
from SteepestDOpti import SteepestDescent, visualisation
from QuasiNwOpti import *

if __name__ == "__main__":
    a1 = Atom("C1", 0.0, 0.0, 0.0, -0.2)
    a2 = Atom("C2", 1.8, 0.3, 0.1,  0.1)
    a3 = Atom("C3", 3.1, 1.0, 0.4,  0.1)
    a4 = Atom("O1", 4.4, 0.5, 1.0, -0.3)

    atoms=[a1,a2,a3,a4]

    terms= [
        Liaison(a1, a2, l0=1.50, k=300),
        Liaison(a2, a3, l0=1.50, k=300),
        Liaison(a3, a4, l0=1.40, k=350),
        Angle(a1, a2, a3,theta0=np.deg2rad(109.5),k0=40),
        Angle(a2, a3, a4,theta0=np.deg2rad(109.5),k0=40),
        TorsionPropre(a1, a2, a3, a4,parametres=[(0.5, 1, 0),(0.2, 3, 0)]),
        VanDerWaals(a1, a4,sigma=3.0, epsilon=0.1),
        Electrostatique( a1, a4,epsilon=80,charge1=a1.charge,charge2=a4.charge)]

    system = System(atoms, terms)

    x0 = system.vecteur_coords() #coords a lorigine
    print(system.energie_totale())
    print(system.energie_par_terme())

    f = system.fontion

    debut_dec=time.perf_counter() # temps de calcule steepest
    coords, norms = SteepestDescent(f,x0,itermax=1000,cutof=0.001)
    fin_dec=time.perf_counter()
    print(f"Temps de calcule Steepest Descent: {fin_dec-debut_dec} secondes")


    x_final = coords[-1] #reprends les x initiaux
    #on remet le system aux coords finales
    system.retour_atoms(x_final)

    print("Coords optimisés")
    print(x_final.reshape(-1, 3))

    print("Norme finale du gradient")
    print(norms[-1])

    print("Nombre d'itérations")
    print(len(coords) - 1)
    visualisation(norms)

#######################################
    print("                  Quasi Newton")
    debut_nw=time.perf_counter() # temps de calcule
    res=quasi_newton(system.fontion,x0,gradient_calc)
    fin_nw=time.perf_counter() # temps de calcule
    print(f"Temps de calcule quasi Newton: {fin_nw-debut_nw} secondes")

    print("Coords optimisés")
    print(res.x.reshape(-1, 3))

    print("Norme finale du gradient")
    print(res.fun)

    print("Nombre d'itérations")
    print(res.nit)
    visualisation_nwtn(res, system.fontion) #moins de pas moins de calculs , BFGS>>>