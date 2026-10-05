from abc import ABC, abstractmethod
import numpy as np
import math
import warnings

from PyInstaller.compat import system
from scipy.constants import R

R=R/4181 #conversion car R en joules

def distance(atm1,atm2):
    dist=np.linalg.norm(atm1.coords-atm2.coords)
    return dist

def angle_valence(a1, a2, a3):
    u,v=a1.coords-a2.coords,a3.coords-a2.coords
    uv=np.dot(u,v)
    u_nrm,v_nrm=np.linalg.norm(u),np.linalg.norm(v)
    cos_theta=uv/(u_nrm*v_nrm)
    cos_theta = np.clip(cos_theta, -1, 1)
    theta=np.arccos(cos_theta)
    return theta

def angle_diedre(a1,a2,a3,a4):
    p1,p2,p3,p4=a1.coords,a2.coords,a3.coords,a4.coords
    v1=p2-p1
    v2=p3-p2
    v3=p4-p3
    norme_plan1=np.cross(v1,v2)
    norme_plan2=np.cross(v2,v3)
    cos_angle=np.dot(norme_plan1,norme_plan2)/(np.linalg.norm(norme_plan1)*np.linalg.norm(norme_plan2))
    cos_angle = np.clip(cos_angle, -1, 1)
    angle_di=np.arccos(cos_angle)
    return angle_di

class Atom:
    def __init__(self,nom,x,y,z,charge):
        self.nom=nom
        self.coords=np.array([x,y,z],dtype=float)
        self.charge=charge

    @property
    def coords(self):
        return self._coords
    @coords.setter
    def coords(self, coords):
        if len(coords)!=3:
            raise ValueError("coords should be 3 elements")
        self._coords=coords

class Termes(ABC):
    @abstractmethod
    def energie(self):
        pass

class Liaison(Termes):
    def __init__(self,a1,a2,l0,k):
        self.l0=l0
        self.k=k
        self.a1=a1
        self.a2=a2

    def energie(self):
        vl=self.k/2*(distance(self.a1,self.a2)-self.l0)**2
        return vl

class Angle(Termes):
    def __init__(self,a1,a2,a3,theta0,k0):
        self.theta0=theta0
        self.a1=a1
        self.a2=a2
        self.a3=a3
        self.k0=k0

    def energie(self):
        v0=self.k0/2*(angle_valence(self.a1,self.a2,self.a3)-self.theta0)**2
        return v0

class TorsionPropre(Termes):
    def __init__(self,a1,a2,a3,a4,parametres):
        self.a1=a1
        self.a2=a2
        self.a3=a3
        self.a4=a4
        self.parametres=parametres
        warnings.warn("Attention : les parametres doivent etre entres par ordre ; Vn,n,gamma")

    def energie(self):
        total=0
        w=angle_diedre(self.a1,self.a2,self.a3,self.a4)
        for vn,n,gamma in self.parametres: #user rentre les parametres sous forme de tuple ou liste
            total=total+ vn/2*(1+np.cos(n*w-gamma)) #rappel : lenergei est la somme pour n parametres
        return total

#classe qui sert de dico des parametres selon latome a implementer

class TorsionImpropre(Termes):
    def __init__(self,a1,a2,a3,a4,w0,kw):
        self.a1=a1
        self.a2=a2
        self.a3=a3
        self.a4=a4
        self.w0=w0
        self.kw=kw
        warnings.warn("Attention : les angles doivent etre en rad")

    def energie(self):
        vw=self.kw/2*(angle_diedre(self.a1,self.a2,self.a3,self.a4)-self.w0)**2
        return vw

class VanDerWaals(Termes):
    def __init__(self,a1,a2,sigma,epsilon):
        self.a1=a1
        self.a2=a2
        self.sigma=sigma
        self.epsilon=epsilon

    def energie(self):
        r=distance(self.a1,self.a2)
        vdw=4*self.epsilon*((self.sigma/r)**12 - (self.sigma/r)**6)
        return vdw

class Electrostatique(Termes):
    F_COULOMB = 332.0637 #conversion car dans la forume de base on a dist en metres et V en joules
#1/4pie0 convertit en coulomb (1.6*10**-19)**2 * 10**10 (conversion de metre en Å) *1/4181 (J en Kcal)

    def __init__(self,a1,a2,epsilon,charge1,charge2):
        self.a1=a1
        self.a2=a2
        self.epsilon=epsilon
        self.charge1=charge1
        self.charge2=charge2

    def energie(self):
        r=distance(self.a1,self.a2)
        ve = self.F_COULOMB * (self.charge1 * self.charge2) / (self.epsilon * r)#conversion permet de remplacer directement par F.coulomb
        return ve

class System:
    def __init__(self,atoms,terms):
        self.atoms=atoms
        self.terms=terms

    def energie_totale(self):
        totale=0
        for term in self.terms:
            totale=totale+term.energie()
        return totale

    def energie_par_terme(self):
        dico={}
        for term in self.terms:
            nom=type(term).__name__
            if nom not in dico:
                dico[nom]=0
            dico[nom]+=term.energie()
        return dico

    def vecteur_coords(self):
        coords=np.array([])
        for atom in self.atoms:
            coords=np.concatenate((coords,atom.coords))
        return coords

    def retour_atoms(self,x):
        nvcoord=np.asarray(x).reshape(-1,3) #evite les erruers de np
        for i in range(np.shape(nvcoord)[0]):
            self.atoms[i].coords=nvcoord[i].copy()

    def fontion(self,x):
        self.retour_atoms(x)
        return self.energie_totale()

def calcul_probabilites(systems,T):
    energies=[]
    for system in systems:
        energies.append(system.energie_totale())

    Z=0
    for i in energies:
        Z+=np.exp(-i/(R*T))

    pi=[]
    for i in energies:
        pi.append(np.exp(-i/(R*T))/Z)

    return pi
