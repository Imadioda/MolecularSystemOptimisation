# MolecularSystemOptimisation

Basic molecular system optimisation by potential energy minimisation (molecular mechanics force field), using two comparable methods:

- **Steepest Descent**, implemented from scratch with a line search;
- **Quasi-Newton BFGS**, through `scipy.optimize.minimize`.

The project lets you define atoms, attach energy terms to them (bonds, angles, torsions, Van der Waals, electrostatics), minimise the total energy and visualise the convergence.

---

## Table of contents

1. [Features](#features)
2. [Requirements and installation](#requirements-and-installation)
3. [Project structure](#project-structure)
4. [Energy model](#energy-model)
5. [How `main_opti.py` works](#how-main_optipy-works)
6. [Adapting the project to your system](#adapting-the-project-to-your-system)
7. [Optimisation algorithms](#optimisation-algorithms)
8. [Units](#units)
9. [Planned updates](#planned-updates)
10. [Known limitations](#known-limitations)

---

## Features

- Atom definition (name, 3D coordinates, partial charge).
- Total energy calculation and **breakdown by term type**.
- Minimisation with **Steepest Descent** (numerical gradient + Wolfe line search).
- Minimisation with **BFGS** (SciPy).
- Convergence plots (gradient norm for Steepest Descent, energy for BFGS).
- Boltzmann probabilities between several systems/conformations at temperature `T`.

---

## Requirements and installation

Python 3.9 or higher is recommended.

Required modules:

| Module | Purpose |
|---|---|
| `numpy` | vector computation, distances, angles |
| `scipy` | physical constants, `line_search`, `minimize` (BFGS) |
| `matplotlib` | convergence plots |

Installation:

```bash
git clone https://github.com/<your-account>/MolecularSystemOptimisation.git
cd MolecularSystemOptimisation
pip install numpy scipy matplotlib
```

Or with a `requirements.txt` file:

```
numpy
scipy
matplotlib
```

```bash
pip install -r requirements.txt
```

> `abc`, `math`, `warnings` and `time` are part of the Python standard library, nothing to install.

---

## Project structure

```
MolecularSystemOptimisation/
├── CalculeEnergie.py   # Atoms, energy terms, System class
├── EnerTools.py        # Numerical gradient, descent direction, optimal step
├── SteepestDOpti.py    # Steepest Descent algorithm + plotting
├── QuasiNwOpti.py      # BFGS wrapper (SciPy) + plotting
├── main_opti.py        # Example script: builds a system and runs both optimisations
└── opti_calcule.py     # Reserved for a future Optimisation class (empty for now)
```

### Role of each file

**`CalculeEnergie.py`**: the core of the model.
- Geometry functions: `distance`, `angle_valence`, `angle_diedre`.
- `Atom`: an atom (`nom`, `coords`, `charge`).
- Energy term classes (all inherit from `Termes` and expose `.energie()`):
  `Liaison`, `Angle`, `TorsionPropre`, `TorsionImpropre`, `VanDerWaals`, `Electrostatique`.
- `System`: groups the list of atoms and the list of terms. Main methods:
  - `energie_totale()`: sum of all terms;
  - `energie_par_terme()`: energy grouped by term type;
  - `vecteur_coords()`: flattens all coordinates into a 1D vector `[x1,y1,z1,x2,...]`;
  - `retour_atoms(x)`: writes a coordinate vector back into the atoms;
  - `fontion(x)`: **the objective function** `f(x)` passed to the optimisers (updates the atoms, then returns the energy).
- `calcul_probabilites(systems, T)`: Boltzmann weights of a list of systems at temperature `T`.

**`EnerTools.py`**: shared numerical tools.
- `gradient_calc(f, x, h=1e-5)`: finite-difference gradient.
- `direction(grad)`: descent direction `-grad`.
- `pas_optimal(f, gradient_calc, x, direction)`: step obtained from `scipy.optimize.line_search`.

**`SteepestDOpti.py`**: `SteepestDescent(f, x0, itermax, cutof)` returns the list of coordinates and gradient norms at each iteration; `visualisation(norms)` plots the convergence.

**`QuasiNwOpti.py`**: `quasi_newton(f, x0, grad_func)` runs BFGS; `visualisation_nwtn(res, f)` plots the energy at each iteration.

---

## Energy model

The total energy is the sum of the terms added to the system:

| Term | Class | Formula | Parameters |
|---|---|---|---|
| Bond stretching | `Liaison` | ½·k·(r − l₀)² | `l0`, `k` |
| Valence angle | `Angle` | ½·k₀·(θ − θ₀)² | `theta0` (rad), `k0` |
| Proper torsion | `TorsionPropre` | Σ Vₙ/2·(1 + cos(nω − γ)) | list of `(Vn, n, gamma)` |
| Improper torsion | `TorsionImpropre` | ½·k_ω·(ω − ω₀)² | `w0` (rad), `kw` |
| Van der Waals | `VanDerWaals` | 4ε·[(σ/r)¹² − (σ/r)⁶] (Lennard-Jones) | `sigma`, `epsilon` |
| Electrostatics | `Electrostatique` | 332.0637·q₁q₂ / (ε·r) | `epsilon`, `charge1`, `charge2` |

---

## How `main_opti.py` works

`main_opti.py` is a **complete example** on a small 4-atom molecule (C1, C2, C3, O1). It goes through these steps:

### 1. Creating the atoms

```python
a1 = Atom("C1", 0.0, 0.0, 0.0, -0.2)   # name, x, y, z (Å), partial charge
a2 = Atom("C2", 1.8, 0.3, 0.1,  0.1)
a3 = Atom("C3", 3.1, 1.0, 0.4,  0.1)
a4 = Atom("O1", 4.4, 0.5, 1.0, -0.3)
atoms = [a1, a2, a3, a4]
```

### 2. Defining the energy terms

```python
terms = [
    Liaison(a1, a2, l0=1.50, k=300),
    Liaison(a2, a3, l0=1.50, k=300),
    Liaison(a3, a4, l0=1.40, k=350),
    Angle(a1, a2, a3, theta0=np.deg2rad(109.5), k0=40),
    Angle(a2, a3, a4, theta0=np.deg2rad(109.5), k0=40),
    TorsionPropre(a1, a2, a3, a4, parametres=[(0.5, 1, 0), (0.2, 3, 0)]),
    VanDerWaals(a1, a4, sigma=3.0, epsilon=0.1),
    Electrostatique(a1, a4, epsilon=80, charge1=a1.charge, charge2=a4.charge),
]
```

Terms reference the `Atom` objects directly: when the atom coordinates change, energies are automatically recomputed with the new positions.

### 3. Building the system and initial energy

```python
system = System(atoms, terms)
x0 = system.vecteur_coords()
print(system.energie_totale())
print(system.energie_par_terme())
```

### 4. Steepest Descent

```python
f = system.fontion
coords, norms = SteepestDescent(f, x0, itermax=1000, cutof=0.001)
```

- `itermax`: maximum number of iterations;
- `cutof`: threshold on the gradient norm used to declare convergence.

The script then puts the system back at the final coordinates (`system.retour_atoms(x_final)`), prints the optimised coordinates, the final gradient norm, the number of iterations and the computation time, then plots the convergence curve.

### 5. Quasi-Newton (BFGS)

```python
res = quasi_newton(system.fontion, x0, gradient_calc)
```

Starting from the **same** `x0`, BFGS returns a comparable result (`res.x`, `res.fun`, `res.nit`) and the energy-per-iteration curve is plotted with `visualisation_nwtn`. The printed computation times let you compare both methods directly.

### Running the example

```bash
python main_opti.py
```

---

## Adapting the project to your system

To use your own molecule, you only need to modify **the construction part** of `main_opti.py`:

1. **Define your atoms**: one `Atom(name, x, y, z, charge)` per atom (coordinates in Å).
2. **List your terms**: add or remove `Liaison`, `Angle`, `TorsionPropre`, `TorsionImpropre`, `VanDerWaals`, `Electrostatique` according to your molecule's topology and your force field parameters.
3. **Create the system**: `system = System(atoms, terms)`.
4. **Choose the optimisation parameters**: `itermax`, `cutof` for Steepest Descent.

Things to keep in mind:

- **Angles are in radians** (use `np.deg2rad(...)` to convert).
- For `TorsionPropre`, parameters are entered in the order **`(Vn, n, gamma)`**, one tuple per harmonic.
- For non-bonded interactions (`VanDerWaals`, `Electrostatique`), add **one instance per atom pair**.
- Charges in `Electrostatique` are passed explicitly; use the atoms' own charges (`a.charge`) to stay consistent.
- To compare several conformations, create several `System` objects and use `calcul_probabilites([sys1, sys2, ...], T)`.

Minimal example with a diatomic molecule:

```python
a = Atom("C", 0.0, 0.0, 0.0, 0.0)
b = Atom("C", 2.0, 0.0, 0.0, 0.0)
system = System([a, b], [Liaison(a, b, l0=1.5, k=300)])

coords, norms = SteepestDescent(system.fontion, system.vecteur_coords(), itermax=500, cutof=1e-3)
system.retour_atoms(coords[-1])
```

---

## Optimisation algorithms

Both methods minimise `f(x) = E(x)`, where `x` is the vector of all coordinates (size 3 × number of atoms).

- **Steepest Descent**: at each iteration, move along `-∇E` with a step determined by `scipy.optimize.line_search` (Wolfe conditions). Simple and robust, but convergence can be slow near the minimum.
- **BFGS**: quasi-Newton method that progressively builds an approximation of the inverse Hessian. It usually needs fewer iterations and fewer energy evaluations.

The gradient is computed **numerically** by finite differences (`h = 1e-5`), which avoids differentiating each term analytically.

---

## Units

| Quantity | Unit |
|---|---|
| Distances | Å |
| Angles | radians |
| Energy | kcal/mol |
| Charges | elementary charge (e) |
| Constant `R` | kcal/(mol·K) (converted from J/(mol·K)) |

---

## Planned updates

A upcoming version will add **PDB** file handling:

- a function that **automatically reads coordinates** (and atom names) from a `.pdb` file to build the `Atom` objects directly, without manual input;
- a function that **automatically writes a `.pdb` file** from the optimised coordinates, so the molecule can be **visualised in PyMOL** (before/after optimisation).

An `Optimisation` class (file `opti_calcule.py`) is also planned to group the optimisation methods under a single interface.

---

## Known limitations

- The project is under development: force field parameters must be supplied by the user.
- The numerical gradient is costly for large systems.
- Dihedral angles are computed in the range [0, π] (unsigned).

---

## License

To be defined (for example MIT).
