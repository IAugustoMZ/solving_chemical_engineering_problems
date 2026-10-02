# Material Balances Module

A comprehensive Python module for modeling and solving material balance problems in chemical engineering processes.

## Overview

This module provides a structured framework for defining chemical process streams, equipment units, and solving material balance equations. It's designed for steady-state material balance calculations in unit operations like evaporators, mixers, splitters, and multi-stage processes.

## Core Classes

### Component

Represents a chemical species or component (e.g., water, acetone, sugar).

```python
from src.material_balances import Component

water = Component("Water", molar_mass=18.015)
acetone = Component("Acetone", molar_mass=58.08)
```

Molar masses are optional for single-basis balances and are required only for
components that participate in a mass/molar conversion. If a balance can be
solved on one basis with partial molar-mass data, the solver reports that basis
and leaves non-derivable results on the other basis blank.

### Stream

Represents a flow of material with specified flow rate and composition.

**Key Features:**
- Keeps the flow-rate basis separate from the composition basis
- Converts mass and molar flow rates and fractions when component molar masses are available
- Allows mixed inputs such as a mass flow rate with mole fractions
- Automatically fills the final missing fraction when the other fractions sum to at most 1
- Validates fractions and input consistency
- Bidirectional (input/output) designation

```python
from src.material_balances import Stream, Component

water = Component("Water", molar_mass=18.015)
acetone = Component("Acetone", molar_mass=58.08)

feed = Stream(
    name="evaporator_feed",
    flow_rate=100.0,
    flow_type="kg/h",
    components=[water, acetone],
    composition={"Water": 0.65, "Acetone": 0.35},
    composition_basis="molar",
)

feed.molar_flow_rate
feed.mass_fractions
```

### ProcessUnit

Represents a piece of equipment with inlet and outlet streams. Performs material balance calculations and degree-of-freedom analysis with support for optional algebraic ratio constraints.

**Capabilities:**
- Validates consistency (inlet and outlet must have same components)
- Calculates independent material balance equations (= number of components)
- Counts unknowns (None flow rates and compositions)
- Incorporates optional ratio constraints to reduce DOF
- Assembles mass- or molar-basis component balances and stream constraints
- Checks structural solvability from the independent constraint rank
- Solves balances with SciPy's `linprog`
- Reconciles every specified flow and fraction before updating streams
- Returns separate mass-basis and molar-basis flow and fraction rows when derivable

```python
from src.material_balances import ProcessUnit, FlowRatio

evaporator = ProcessUnit(
    name="Evaporator",
    input_streams=[feed],
    output_streams=[vapor, liquid],
    ratios=[]  # Optional list of Ratio constraints
)

if evaporator.is_solvable():
    evaporator.solve_material_balances()
    evaporator.print_report()
```

## Solver Functions

### calculate_stream_based_on_jam_mass()

Solves a specific material balance problem: strawberry jam production.

Given desired jam mass, calculates required strawberries, sugar, and water evaporated.

```python
from src.material_balances import calculate_stream_based_on_jam_mass

result = calculate_stream_based_on_jam_mass(1000.0)  # 1000 kg jam desired
print(f"Strawberries: {result['strawberry_mass']:.1f} kg")
print(f"Sugar: {result['sugar_mass']:.1f} kg")
print(f"Water evaporated: {result['water_mass']:.1f} kg")
```

## Usage Example: Acetone-Water Evaporator

```python
from src.material_balances import Component, Stream, ProcessUnit

# Create components
water = Component("Water")
acetone = Component("Acetone")

# Create streams
feed = Stream(
    "feed", 10.0, "mole",
    [water, acetone],
    {"Water": 0.65, "Acetone": None}
)

vapor = Stream(
    "vapor", None, "mole",
    [water, acetone],
    {"Water": 0.75, "Acetone": None}
)

liquid = Stream(
    "liquid", None, "mole",
    [water, acetone],
    {"Water": 0.187, "Acetone": None}
)

# Create and solve evaporator
evaporator = ProcessUnit("Evaporator", [feed], [vapor, liquid])

print(f"Solvable: {evaporator.is_solvable()}")
print(f"Unknowns: {evaporator.unknowns}")
print(f"Independent material balances: {evaporator.independent_material_balances}")

evaporator.solve_material_balances()
evaporator.print_report()
```

### Degree-of-freedom reports

`report_degrees_of_freedom()` prints each unknown stream field, every
component-flow variable (including the selected mass or molar basis), and a
rank-increasing list of independent constraints. This makes it clear which
stream specifications determine the reported structural rank.

```text
Unknown input fields (flows + fractions): 2
  - Stream 'CarryGas': flow rate
  - Stream 'Mixture': flow rate
Component-flow variables (molar basis): 6
  1. Stream 'CarryGas': 'CO2'
  2. Stream 'CarryGas': 'others'
Independent constraint rank: 6
```

## Mass and Molar Basis Handling

Use `molar_mass` on each component and set `composition_basis` independently
from `flow_type`. Flow units and molar-mass units must be compatible. For
example, `kg/h` with molar masses in `kg/kmol` yields `kmol/h`.

```python
from src.material_balances import StreamFactory

factory = StreamFactory(
    ["Methanol", "Water"],
    default_flow_type="kg/h",
    molar_masses={"Methanol": 32.04, "Water": 18.015},
)
feed = factory.add_stream(
    "Feed", [0.5, 0.5], flow_rate=100.0, composition_basis="molar"
)
product = factory.add_stream(
    "Product", [0.5, 0.5], flow_type="kmol/h",
    composition_basis="molar", direction="output",
)
unit = factory.build_process_unit("Transfer")
results = unit.solve_material_balances()
```

`molar_masses` may also be a positional list aligned with the component names;
use `None` for a component whose molar mass is unavailable:

```python
factory = StreamFactory(
    ["CO2", "Carrier gas"],
    default_flow_type="kmol/min",
    molar_masses=[44.01, None],
)
```

The returned dataframe separates every flow into `mass_flow_rate` or
`molar_flow_rate`, and every component flow into
`component_mass_flow_rate` or `component_molar_flow_rate`. It likewise uses
`mass_fraction` and `mole_fraction`; there are no native-basis rows that mix
mass and molar values across streams. A derived value is blank when its
conversion cannot be determined from the supplied molar masses. For example,
a pure CO2 tracer specified in kg/min can be converted and used in a molar
balance with only CO2's molar mass; total mass flows remain unavailable when
the carrier-gas molar mass is not supplied.

Ratio constraints that compare total or component flow rates can accept
`basis="mass"` or `basis="molar"`. If omitted, flow ratios use the process
unit's selected balance basis; component-flow values use the referenced
stream's flow basis.

## Continuous integration

The GitHub Actions quality and test workflows run for pull requests targeting
`main`. They install the Poetry environment, then check Black formatting,
isort imports, selected Flake8 errors, Pylint errors, the full pytest suite,
and test coverage.

## Ratio Constraints

The module supports optional algebraic constraints that represent independent relationships between streams or components, reducing the system's degrees of freedom.

### Available Constraint Types

#### FlowRatio
Constrains the ratio of flow rates between two streams.

```python
from src.material_balances import FlowRatio

ratio = FlowRatio(
    stream1_name="inlet1",
    stream2_name="inlet2",
    target_ratio=0.8  # inlet1.F / inlet2.F = 0.8
)
```

#### ComponentFlowRatio
Constrains the ratio of component flows between two streams.

```python
from src.material_balances import ComponentFlowRatio

ratio = ComponentFlowRatio(
    stream1_name="Strawberry",
    comp1_name="Solids",
    stream2_name="Sugar",
    comp2_name="Sugar",
    target_ratio=0.45/0.55
)
# (Strawberry.F × Strawberry.x_solids) / (Sugar.F × Sugar.x_sugar) = 45/55
```

#### CompositionRatio
Constrains the ratio of component compositions within a single stream.

```python
from src.material_balances import CompositionRatio

ratio = CompositionRatio(
    stream_name="outlet",
    comp1_name="Acetone",
    comp2_name="Water",
    target_ratio=3.0  # x_acetone / x_water = 3.0
)
```

### Usage Example: Jam Production with Ratio Constraint

```python
from src.material_balances import (
    Component, Stream, ProcessUnit, FlowRatio
)

# Components
strawberry_solids = Component("Solids")
sugar = Component("Sugar")
water = Component("Water")

# Input streams
strawberry = Stream(
    "Strawberry", None, "mass",
    [strawberry_solids, sugar, water],
    {"Solids": 0.15, "Sugar": 0, "Water": None}
)

sugar_inlet = Stream(
    "Sugar", None, "mass",
    [strawberry_solids, sugar, water],
    {"Solids": 0, "Sugar": 1.0, "Water": 0}
)

# Output streams
jam = Stream(
    "Jam", 1.0, "mass",
    [strawberry_solids, sugar, water],
    {"Solids": 0.1, "Sugar": 0.567, "Water": None}
)

water_evap = Stream(
    "Evaporated", None, "mass",
    [strawberry_solids, sugar, water],
    {"Solids": 0, "Sugar": 0, "Water": 1.0}
)

# Ratio constraint: strawberry/sugar = 45/55
ratio = FlowRatio("Strawberry", "Sugar", target_ratio=45/55)

# Solve with constraint
heater = ProcessUnit(
    "Heater",
    [strawberry, sugar_inlet],
    [jam, water_evap],
    ratios=[ratio]  # Add ratio constraint
)

print(f"Solvable: {heater.is_solvable()}")
heater.solve_material_balances()
heater.print_report()
```

## Degree of Freedom Analysis (with Ratio Constraints)

Solvability is assessed from the rank of the assembled component-flow
constraints. The system is structurally determined when this rank equals the
number of component-flow variables. Feasibility and consistency of all supplied
values are checked by `solve_material_balances()`.

## Validation and Error Handling

The module includes extensive validation:

1. **Stream Composition**: Must sum to 1.0 when all specified
2. **Composition Basis**: Must be `mass` or `molar` for material balances
3. **Molar Mass**: Must be positive and supplied for components whose mass/molar conversion is needed
4. **Stream Consistency**: All inlet and outlet streams must contain identical components
5. **Solvability**: Constraints must determine a unique nonnegative component-flow solution

All errors raise informative exceptions with clear messages.

## Solver Details

Material balance solving uses SciPy's `linprog` with component flow rates as
nonnegative decision variables. Material balances, known stream flows and
compositions, and supported ratio constraints are represented as linear
equalities. This removes the need for nonlinear initial guesses. The solver
rejects infeasible or non-unique systems, checks balance residuals before
updating streams, and uses a default absolute material-balance tolerance of
1e-6.

## Test Suite

Comprehensive test coverage (51 tests across two test files):

### test_material_balances.py (26 tests)
- **Component tests**: Basic creation and attributes
- **Stream tests**: Initialization, validation, complementary composition, edge cases
- **ProcessUnit tests**: Initialization, DOF analysis, solvability, solving, multi-stream cases
- **Integration tests**: Complete problem workflows

### test_ratio_constraints.py (25 tests)
- **FlowRatio tests**: Creation, residual calculation, validation
- **ComponentFlowRatio tests**: Component flow calculations, multi-component systems
- **CompositionRatio tests**: Composition ratio verification
- **ProcessUnit+Ratios tests**: Integration with ratio constraints, DOF accounting
- **Jam production case**: Real-world example with flow ratio constraint

Run tests with:
```bash
# Run all tests
poetry run pytest tests/ -v

# Run specific test file
poetry run pytest tests/test_ratio_constraints.py -v

# Run with coverage
poetry run pytest tests/ --cov=src --cov-report=term
```

## Requirements

- Python 3.14+
- numpy >= 2.5.3
- scipy >= 1.18.1

## Module Structure

```
src/material_balances/
├── __init__.py                  # Public API
├── components.py                # Component class
├── stream.py                    # Stream class
├── process_unit.py              # ProcessUnit class with ratio support
├── ratio_constraints.py         # Ratio constraint classes (Ratio, FlowRatio, etc.)
├── README.md                    # This file
└── solvers.py                   # Solver functions

tests/
├── test_material_balances.py    # Stream, Component, ProcessUnit tests (26 tests)
└── test_ratio_constraints.py    # Ratio constraint integration tests (25 tests)
```
