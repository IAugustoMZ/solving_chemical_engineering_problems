# Solving Chemical Engineering Problems 🧪⚗️

A collection of computational solutions, modeling workflows, and interactive Jupyter notebooks for classic and modern **Chemical Engineering** problems using **Python 3.14**.

---

## 🎯 Purpose

This project is a dedicated space to explore, simulate, and solve chemical engineering problems with modern scientific computing tools. Rather than relying solely on black-box commercial software, the goal is to:
- Deepen understanding of fundamental engineering principles from first principles to numerical simulations.
- Build transparent, reproducible, and interactive notebooks for unit operations, transport phenomena, thermodynamics, kinetics, and process control.
- Leverage modern Python tools for equation solving, parameter optimization, symbolic math, and physical unit tracking.

---

## 📚 Topics & Roadmap

The repository is structured around core Chemical Engineering domains:

| Topic | Description / Focus Areas |
| :--- | :--- |
| **00. Mass & Energy Balances** | Single-unit and multi-unit material balances, degree of freedom (DOF) analysis, linear solvers for non-reactive steady-state systems, accumulation problems. |
| **01. Thermodynamics** | Equations of State (Peng-Robinson, SRK, Van der Waals), phase equilibria (VLE, LLE, VLLE), flash calculations, fugacity, chemical reaction equilibrium. |
| **02. Fluid Mechanics** | Friction factor calculations, pipe networks, pressure drop in packed beds (Ergun equation), pump sizing, non-Newtonian fluids. |
| **03. Heat Transfer** | Conduction (1D/2D steady and transient), convective heat transfer coefficients, heat exchanger sizing and rating (LMTD & $\varepsilon$-NTU methods), radiation. |
| **04. Mass Transfer & Separations** | Diffusion and mass transfer coefficients, distillation (McCabe-Thiele, Ponchon-Savarit, Fenske-Underwood-Gilliland), gas absorption, liquid-liquid extraction, membrane separation. |
| **05. Chemical Reaction Engineering** | Ideal reactors (Batch, CSTR, PFR), reactor sizing, non-isothermal & non-adiabatic operation, multiple reactions and selectivity, catalytic reactor modeling. |
| **06. Process Dynamics & Control** | Dynamic system modeling, transfer functions, step responses, feedback control (PID tuning), frequency response, and stability analysis. |
| **07. Numerical Methods** | Non-linear root finding, system of ODE/PDE integration, parameter estimation & curve fitting, constrained optimization. |

---

## 🛠️ Tech Stack & Dependencies

- **Python**: `^3.14`
- **Dependency Management**: [Poetry](https://python-poetry.org/)
- **Core Libraries**:
  - [`NumPy`](https://numpy.org/) — High-performance array operations and linear algebra
  - [`SciPy`](https://scipy.org/) — Numerical integration (`solve_ivp`), optimization, and root finding (`fsolve`, `root`)
  - [`Matplotlib`](https://matplotlib.org/) — Engineering plotting and visualization
  - [`Pandas`](https://pandas.pydata.org/) — Tabular data handling, experimental datasets, and simulation outputs
  - [`SymPy`](https://www.sympy.org/) — Symbolic mathematics, analytical derivatives, and exact solutions
  - [`Pint`](https://pint.readthedocs.io/) — Physical units, dimensional analysis, and automated unit conversions
  - [`ipykernel`](https://github.com/ipython/ipykernel) — Jupyter execution kernel

---

## ✨ Features Implemented

### Material Balances (`src.material_balances`)
- **Multistream Material Balance Solver**: A generalized robust solver for steady-state, non-reactive multi-stream and multi-component systems.
  - Automatically performs Degree of Freedom (DOF) analysis accounting for ratio constraints.
  - Identifies unknown flow rates and species mass/mole fractions.
  - Uses scipy's `least_squares` optimizer with non-negative constraints for robust numerical solution.
  - Prints structured reports with solved stream variables, flows, and diagnostic messages.

- **Ratio Constraints**: Flexible algebraic constraint system for process design with independent relationships:
  - **FlowRatio**: Enforce relationships between stream flow rates (e.g., `F1 / F2 = k`)
  - **ComponentFlowRatio**: Constrain component flow rate ratios across streams
  - **CompositionRatio**: Specify composition ratios within a stream
  - Each independent ratio reduces system DOF by 1, enabling unique solutions for previously underdetermined problems
  - Example: Jam production case with strawberry-to-sugar mass ratio of 45:55

- **Binary Mixture Separations**: Specific tools for solving standard vapor-liquid separation tasks.
- **Accumulation Problems**: Example solvers for liquid tank filling via integral balances.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.14** installed on your system.
- **Poetry** (version 2.x recommended).

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/IAugustoMZ/solving_chemical_engineering_problems.git
   cd solving_chemical_engineering_problems
   ```

2. **Install dependencies with Poetry**:
   ```bash
   poetry install
   ```

3. **Register the Poetry environment as a Jupyter kernel** (optional but recommended):
   ```bash
   poetry run python -m ipykernel install --user --name solving-chem-eng --display-name "Python 3.14 (ChemEng)"
   ```

4. **Launch Jupyter Lab / Notebook**:
   ```bash
   poetry run jupyter lab
   # or
   poetry run jupyter notebook
   ```

---

## 📁 Repository Structure

```text
solving_chemical_engineering_problems/
├── pyproject.toml
├── poetry.lock
├── README.md
├── .gitignore
├── src/
│   ├── utils/
│   │   └── validators.py
│   └── material_balances/
│       ├── models/
│       │   ├── stream.py
│       │   └── balance_system.py
│       └── simple_material_balances.py
├── 00_mass_energy_balances/
└── notebooks/
    ├── 01_thermodynamics/
    ├── 02_fluid_mechanics/
    ├── 03_heat_transfer/
    ├── 04_mass_transfer_and_separations/
    ├── 05_reaction_engineering/
    ├── 06_process_control/
    └── 07_numerical_methods/
```

---

## 🧪 Testing & Quality Checks

This repository maintains high code quality standards with automated testing and linting checks.

### Running Tests

```bash
# Run all tests with coverage
poetry run pytest tests/ -v --cov=src --cov-report=term

# Run specific test file
poetry run pytest tests/test_material_balances.py -v
```

### Code Quality Checks

The project uses **black** (formatting), **isort** (import sorting), **flake8** (linting), and **pylint** (code analysis).

#### Local Quality Checks

```bash
# Run quality checks locally before pushing
./scripts/quality-check.sh

# Run full pre-push checks (tests + quality)
./scripts/pre-push.sh
```

#### Automatic Fixes

```bash
# Auto-format code with black
poetry run black src/ tests/

# Auto-sort imports with isort
poetry run isort src/ tests/
```

### GitHub Actions

Quality checks automatically run on:
- Every push to `main` or `feature-*` branches
- All pull requests to `main`

The workflow checks:
- ✅ Code formatting (black)
- ✅ Import sorting (isort)
- ✅ Linting (flake8, pylint)
- ✅ Tests with coverage
- ⚠️ Duplicate code detection

---

## 💡 Contributing & Adding Problems

Feel free to open issues or submit pull requests with interesting problems, textbook examples (e.g., Fogler, Levenspiel, Smith-Van Ness-Abbott, Geankoplis, Incropera), or real-world process simulations!

**Before submitting a PR:**
1. Ensure all tests pass: `poetry run pytest tests/`
2. Run quality checks: `./scripts/quality-check.sh`
3. Update documentation as needed
