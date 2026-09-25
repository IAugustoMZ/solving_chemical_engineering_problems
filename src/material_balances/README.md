# Material Balances Module

A comprehensive Python module for modeling and solving material balance problems in chemical engineering processes.

## Overview

This module provides a structured framework for defining chemical process streams, equipment units, and solving material balance equations. It's designed for steady-state material balance calculations in unit operations like evaporators, mixers, splitters, and multi-stage processes.

## Core Classes

### Component

Represents a chemical species or component (e.g., water, acetone, sugar).

```python
from src.material_balances import Component

water = Component("Water")
acetone = Component("Acetone")
```

### Stream

Represents a flow of material with specified flow rate and composition.

**Key Features:**
- Supports mass, molar, or volumetric flow rates
- Automatic calculation of missing composition values (when exactly one is unknown)
- Validates that composition fractions sum to 1.0
- Bidirectional (input/output) designation

```python
from src.material_balances import Stream, Component

water = Component("Water")
acetone = Component("Acetone")

feed = Stream(
    name="evaporator_feed",
    flow_rate=10.0,  # Can be None if unknown
    flow_type="mole",
    components=[water, acetone],
    composition={"Water": 0.65, "Acetone": None}  # Automatically calculates Acetone = 0.35
)
```

### ProcessUnit

Represents a piece of equipment with inlet and outlet streams. Performs material balance calculations and degree-of-freedom analysis with support for optional algebraic ratio constraints.

**Capabilities:**
- Validates consistency (inlet and outlet must have same components)
- Calculates independent material balance equations (= number of components)
- Counts unknowns (None flow rates and compositions)
- Incorporates optional ratio constraints to reduce DOF
- Determines if system is solvable (unknowns ≤ equations + ratios)
- Solves material balances using scipy's least_squares optimizer
- Generates detailed balance reports with residual checks

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

The module automatically performs DOF analysis accounting for all constraints:

**Degrees of Freedom = Unknowns - Independent Material Balance Equations - Number of Independent Ratios**

Where:
- **Independent Material Balance Equations** = Number of unique components
- **Unknowns** = Number of None values in all stream flow rates and compositions
- **Number of Independent Ratios** = Count of ratio constraints (each reduces DOF by 1)

System is **solvable** when: **DOF ≤ 0** (i.e., unknowns ≤ equations + ratios)

### Example: Splitter with Flow Ratio

- Single component: 1 independent material balance
- Two output streams with unknown flow rates: 2 unknowns
- Without ratio: DOF = 2 - 1 = 1 → Not solvable ✗
- **With FlowRatio constraint**: DOF = 2 - 1 - 1 = 0 → Solvable ✓

## Validation and Error Handling

The module includes extensive validation:

1. **Stream Composition**: Must sum to 1.0 when all specified
2. **Component Mismatch**: At most one composition value per stream can be None
3. **Stream Consistency**: All inlet and outlet streams must contain identical components
4. **Solvability**: System must have enough equations for unknowns

All errors raise informative exceptions with clear messages.

## Solver Details

Material balance solving uses scipy's `least_squares` optimizer:

- **Objective**: Minimize sum of squared residuals from material balance equations
- **Constraints**: All flow rates and compositions constrained to [0, ∞]
- **Convergence**: Automatically checks success and reports failures
- **Tolerance**: Default mass balance residual tolerance = 1e-6

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
