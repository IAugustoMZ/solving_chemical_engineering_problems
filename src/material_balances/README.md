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

Represents a piece of equipment with inlet and outlet streams. Performs material balance calculations and degree-of-freedom analysis.

**Capabilities:**
- Validates consistency (inlet and outlet must have same components)
- Calculates independent material balance equations (= number of components)
- Counts unknowns (None flow rates and compositions)
- Determines if system is solvable (unknowns ≤ equations)
- Solves material balances using scipy's least_squares optimizer
- Generates detailed balance reports with residual checks

```python
from src.material_balances import ProcessUnit

evaporator = ProcessUnit(
    name="Evaporator",
    input_streams=[feed],
    output_streams=[vapor, liquid]
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

## Degree of Freedom Analysis

The module automatically performs DOF analysis:

**Degrees of Freedom = Unknowns - Independent Material Balance Equations**

Where:
- **Independent Material Balance Equations** = Number of unique components
- **Unknowns** = Number of None values in all stream flow rates and compositions

System is **solvable** when: **DOF ≤ 0** (i.e., unknowns ≤ equations)

Example:
- Evaporator with 2 components: 2 independent material balances
- 2 output streams with unknown flow rates: 2 unknowns
- DOF = 2 - 2 = 0 → Solvable ✓

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

Comprehensive test coverage (36 tests):

- **Component tests**: Basic creation and attributes
- **Stream tests**: Initialization, validation, complementary composition, edge cases
- **ProcessUnit tests**: Initialization, DOF analysis, solvability, solving, multi-stream cases
- **Solver tests**: Jam calculation, mass balance verification, parameter sensitivity
- **Integration tests**: Complete problem workflows

Run tests with:
```bash
poetry run pytest tests/test_material_balances.py -v
```

## Requirements

- Python 3.14+
- numpy >= 2.5.3
- scipy >= 1.18.1

## Module Structure

```
src/material_balances/
├── __init__.py              # Public API
├── components.py            # Component class
├── stream.py                # Stream class
├── process_unit.py          # ProcessUnit class
├── solvers.py               # Solver functions
└── README.md                # This file

tests/
└── test_material_balances.py  # Comprehensive test suite
```
