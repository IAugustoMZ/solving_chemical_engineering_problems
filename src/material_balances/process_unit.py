"""
ProcessUnit class for modeling process units in material balance problems.

This module provides the ProcessUnit class for setting up and solving
material balance problems around process units with inlet and outlet streams.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .ratio_constraints import (
    ComponentFlowRatio,
    ComponentFlowValue,
    CompositionRatio,
    FlowRatio,
)


class ProcessUnit:
    """
    Represents a process unit with inlet and outlet streams.

    A ProcessUnit encapsulates a single piece of equipment or process step
    (e.g., evaporator, mixer, reactor) and performs material balance calculations
    across it. It validates stream consistency, performs degree-of-freedom analysis,
    and solves for unknown flow rates and compositions.

    The number of independent material balance equations equals the number of
    components. Degrees-of-freedom checks assess structural determination; actual
    feasibility is checked by the solver.

    Attributes:
        name (str): Unit identifier or tag.
        input_streams (list[Stream]): List of inlet streams.
        output_streams (list[Stream]): List of outlet streams.
        tol (float): Tolerance for material balance residuals (default 1e-6).
        independent_material_balances (int): Number of component balance equations.
        unknowns (int): Total number of unknown values to solve for.
    """

    def __init__(
        self, name: str, input_streams: list, output_streams: list, ratios: list = None
    ) -> None:
        """
        Initialize a ProcessUnit with inlet and outlet streams.

        Validates that all input and output streams contain the same set of
        components. Calculates the number of independent material balances
        and counts unknowns.

        Parameters:
            name (str): Unit identifier or tag.
            input_streams (list[Stream]): List of inlet Stream objects.
            output_streams (list[Stream]): List of outlet Stream objects.
            ratios (list, optional): List of Ratio constraint objects that represent
                                     independent relationships between streams/components.
                                     Each ratio reduces DOF by 1. Defaults to empty list.

        Raises:
            ValueError: If inlet and outlet streams don't have the same components,
                       or if ratio references are invalid.

        Example:
            >>> water = Component("Water")
            >>> acetone = Component("Acetone")
            >>> feed = Stream("feed", 10, "mole", [water, acetone],
            ...               {"Water": 0.65, "Acetone": None})
            >>> vapor = Stream("vapor", None, "mole", [water, acetone],
            ...                {"Water": 0.25, "Acetone": None})
            >>> liquid = Stream("liquid", None, "mole", [water, acetone],
            ...                 {"Water": 0.813, "Acetone": None})
            >>> evap = ProcessUnit("Evaporator", [feed], [vapor, liquid])
        """
        self.name = name
        self.tol = 1e-6
        self.input_streams = input_streams
        self.output_streams = output_streams
        self.ratios = ratios or []
        self._validate_stream_components()
        self._select_solver_basis()
        self._validate_ratio_references()
        self.calculate_independent_material_balances()
        self.calculate_unknowns()

    def _validate_stream_components(self) -> None:
        """
        Validate that all inlet and outlet streams have the same components.

        Raises:
            ValueError: If inlet and outlet stream components don't match.
        """
        input_components = {
            comp.name for stream in self.input_streams for comp in stream.components
        }
        output_components = {
            comp.name for stream in self.output_streams for comp in stream.components
        }
        if input_components != output_components:
            raise ValueError("Input and output streams must have the same components.")
        for stream in self.input_streams + self.output_streams:
            stream_components = {component.name for component in stream.components}
            if stream_components != input_components:
                raise ValueError(
                    "Every stream in the process unit must contain the same components."
                )
        molar_masses = {}
        for stream in self.input_streams + self.output_streams:
            for component in stream.components:
                if component.name in molar_masses:
                    known = molar_masses[component.name]
                    if known is not None and component.molar_mass is not None and not np.isclose(known, component.molar_mass):
                        raise ValueError(
                            f"Component '{component.name}' has inconsistent molar masses across streams."
                        )
                    if known is None:
                        molar_masses[component.name] = component.molar_mass
                else:
                    molar_masses[component.name] = component.molar_mass

    def _select_solver_basis(self) -> None:
        """Choose mass or molar component flows as the common balance variables."""
        streams = self.input_streams + self.output_streams
        if any(stream.flow_basis not in ("mass", "molar") for stream in streams):
            raise ValueError("Process-unit flow types must use mass or molar units.")
        bases = {stream.flow_basis for stream in streams}
        composition_bases = {stream.composition_basis for stream in streams}
        needs_conversion = len(bases) > 1 or any(
            stream.flow_basis != stream.composition_basis for stream in streams
        )
        if not needs_conversion and len(composition_bases) == 1:
            self.solver_basis = next(iter(bases))
        else:
            self.solver_basis = "molar"
        if self.solver_basis == "molar" and needs_conversion:
            missing = [
                component.name
                for stream in streams
                for component in stream.components
                if component.molar_mass is None
            ]
            if missing:
                raise ValueError(
                    "Molar masses are required to reconcile mass and molar data; "
                    f"missing for: {', '.join(sorted(set(missing)))}."
                )
        if self.solver_basis == "molar" and len(bases) > 1 and any(
            component.molar_mass is None
            for stream in streams
            for component in stream.components
        ):
            raise ValueError(
                "Molar masses are required when a process unit mixes mass and molar flows."
            )
        for ratio in self.ratios:
            if isinstance(ratio, (FlowRatio, ComponentFlowRatio)):
                ratio.resolved_basis = ratio.basis or self.solver_basis
            elif isinstance(ratio, ComponentFlowValue):
                stream = next(
                    (s for s in streams if s.name == ratio.stream_name), None
                )
                ratio.resolved_basis = ratio.basis or (
                    stream.flow_basis if stream is not None else self.solver_basis
                )

    def _validate_ratio_references(self) -> None:
        """
        Validate that all ratio constraints reference valid streams and components.

        Raises:
            ValueError: If a ratio references invalid streams or components.
        """
        streams_dict = {
            stream.name: stream for stream in self.input_streams + self.output_streams
        }

        for i, ratio in enumerate(self.ratios):
            try:
                ratio.validate_references(streams_dict)
            except (KeyError, ValueError) as e:
                raise ValueError(
                    f"Ratio {i} ({ratio.description}) has invalid references: {e}"
                )

    def calculate_independent_material_balances(self) -> None:
        """
        Calculate the number of independent material balance equations.

        The number of independent material balances equals the number of
        unique components in the system.
        """
        self.independent_material_balances = max(
            len(stream.components)
            for stream in self.input_streams + self.output_streams
        )

    def calculate_unknowns(self) -> None:
        """
        Count the total number of unknowns in the system.

        Unknowns include: None flow_rate values and None composition values
        across all inlet and outlet streams.
        """
        self.unknowns = sum(
            int(stream.flow_rate is None)
            + list(stream.composition.values()).count(None)
            for stream in self.input_streams + self.output_streams
        )

    def is_solvable(self) -> bool:
        """Return whether the assembled component-flow constraints determine a unique solution.

        This rank check assesses structural determination. The solver checks
        nonnegativity and consistency of the specified values.
        """
        streams = self.input_streams + self.output_streams
        names = sorted(
            {component.name for stream in streams for component in stream.components}
        )
        _, matrix, _, _ = self._build_component_flow_system(names)
        row_scales = np.maximum(np.max(np.abs(matrix), axis=1), 1.0)
        rank = np.linalg.matrix_rank(matrix / row_scales[:, np.newaxis])
        return rank == len(streams) * len(names)

    def report_degrees_of_freedom(self) -> None:
        """Report structural degrees of freedom using the assembled constraints."""
        streams = self.input_streams + self.output_streams
        component_names = sorted(
            {component.name for stream in streams for component in stream.components}
        )
        _, matrix, _, _ = self._build_component_flow_system(component_names)
        variables = len(streams) * len(component_names)
        row_scales = np.maximum(np.max(np.abs(matrix), axis=1), 1.0)
        rank = np.linalg.matrix_rank(matrix / row_scales[:, np.newaxis])
        dof = variables - rank

        print(f"\nDegrees of Freedom Analysis for '{self.name}':")
        print(f"  Unknown input fields (flows + fractions): {self.unknowns}")
        print(f"  Component-flow variables: {variables}")
        print(f"  Independent constraint rank: {rank}")
        print(f"  Structural degrees of freedom: {dof}")
        if dof > 0:
            print(f"  Status: UNDERDETERMINED (need {dof} independent constraint(s))")
        else:
            print("  Status: STRUCTURALLY DETERMINED (feasibility is checked when solving)")

    def suggest_missing_information(self) -> None:
        """Print missing stream values when the assembled constraints are underdetermined."""
        if self.is_solvable():
            print(
                f"Process unit '{self.name}' has zero structural degrees of freedom. "
                "Run solve_material_balances() to check feasibility."
            )
            return
        print("The process unit is structurally underdetermined. Possible missing values:")
        for stream in self.input_streams + self.output_streams:
            if stream.flow_rate is None:
                print(f"  - Stream '{stream.name}': flow rate")
            for component_name, value in stream.composition.items():
                if value is None:
                    print(f"  - Stream '{stream.name}': composition for '{component_name}'")
        print("Some listed values may be dependent; provide enough independent information.")

    def _collect_unknowns(self) -> list:
        """
        Collect references to all unknown values in the system.

        Returns:
            list: List of tuples (stream, kind, key) for each unknown.
                 kind is "flow_rate" or "composition".
        """
        unknown_refs = []
        for stream in self.input_streams + self.output_streams:
            if stream.flow_rate is None:
                unknown_refs.append((stream, "flow_rate", None))
            for comp_name, comp_value in stream.composition.items():
                if comp_value is None:
                    unknown_refs.append((stream, "composition", comp_name))
        return unknown_refs

    def _build_component_flow_system(self, component_names: list):
        """Build linear equalities using a common mass or molar component-flow basis."""
        streams = self.input_streams + self.output_streams
        variable_refs = [(stream, name) for stream in streams for name in component_names]
        variable_index = {ref: i for i, ref in enumerate(variable_refs)}
        rows, targets, labels = [], [], []

        def add_equation(coefficients, target, label):
            row = np.zeros(len(variable_refs), dtype=float)
            for ref, coefficient in coefficients.items():
                row[variable_index[ref]] += coefficient
            rows.append(row)
            targets.append(float(target))
            labels.append(label)

        def molecular_weight(stream, component_name):
            return next(c.molar_mass for c in stream.components if c.name == component_name)

        def flow_conversion(stream, component_name, basis):
            if self.solver_basis == basis:
                return 1.0
            mw = molecular_weight(stream, component_name)
            if mw is None:
                raise ValueError(
                    f"Molar mass for '{component_name}' is required to convert flow bases."
                )
            return mw if basis == "mass" else 1.0 / mw

        def fraction_coefficients(stream, comp_name, fraction):
            # The specified fraction is the selected component amount divided by
            # the total amount on its own basis.
            coeff = {}
            for other_name in component_names:
                factor = flow_conversion(stream, other_name, stream.composition_basis)
                coeff[(stream, other_name)] = -fraction * factor
            coeff[(stream, comp_name)] += flow_conversion(
                stream, comp_name, stream.composition_basis
            )
            return coeff

        for comp_name in component_names:
            coeff = {(stream, comp_name): 1.0 for stream in self.input_streams}
            coeff.update({(stream, comp_name): -1.0 for stream in self.output_streams})
            add_equation(coeff, 0.0, f"material balance for {comp_name}")

        for stream in streams:
            if stream.flow_rate is not None:
                add_equation(
                    {
                        (stream, name): flow_conversion(stream, name, stream.flow_basis)
                        for name in component_names
                    },
                    stream.flow_rate,
                    f"total flow for {stream.name}",
                )
            for name, fraction in stream.composition.items():
                if fraction is not None:
                    add_equation(
                        fraction_coefficients(stream, name, fraction),
                        0.0,
                        f"{name} {stream.composition_basis} fraction for {stream.name}",
                    )

        streams_by_name = {stream.name: stream for stream in streams}
        for ratio in self.ratios:
            if isinstance(ratio, FlowRatio):
                first = streams_by_name[ratio.stream1_name]
                second = streams_by_name[ratio.stream2_name]
                basis = getattr(ratio, "basis", None) or self.solver_basis
                if basis not in ("mass", "molar"):
                    raise ValueError("FlowRatio basis must be 'mass' or 'molar'.")
                coeff = {
                    (first, name): flow_conversion(first, name, basis)
                    for name in component_names
                }
                for name in component_names:
                    ref = (second, name)
                    coeff[ref] = coeff.get(ref, 0.0) - ratio.target_ratio * flow_conversion(
                        second, name, basis
                    )
                add_equation(coeff, 0.0, ratio.description)
            elif isinstance(ratio, CompositionRatio):
                stream = streams_by_name[ratio.stream_name]
                basis = stream.composition_basis
                coeff = {
                    (stream, ratio.comp1_name): flow_conversion(
                        stream, ratio.comp1_name, basis
                    )
                }
                ref = (stream, ratio.comp2_name)
                coeff[ref] = coeff.get(ref, 0.0) - ratio.target_ratio * flow_conversion(
                    stream, ratio.comp2_name, basis
                )
                add_equation(coeff, 0.0, ratio.description)
            elif isinstance(ratio, ComponentFlowRatio):
                first = streams_by_name[ratio.stream1_name]
                second = streams_by_name[ratio.stream2_name]
                basis = getattr(ratio, "basis", None) or self.solver_basis
                if basis not in ("mass", "molar"):
                    raise ValueError("ComponentFlowRatio basis must be 'mass' or 'molar'.")
                coeff = {
                    (first, ratio.comp1_name): flow_conversion(
                        first, ratio.comp1_name, basis
                    )
                }
                ref = (second, ratio.comp2_name)
                coeff[ref] = coeff.get(ref, 0.0) - ratio.target_ratio * flow_conversion(
                    second, ratio.comp2_name, basis
                )
                add_equation(coeff, 0.0, ratio.description)
            elif isinstance(ratio, ComponentFlowValue):
                stream = streams_by_name[ratio.stream_name]
                basis = getattr(ratio, "basis", None) or stream.flow_basis
                if basis not in ("mass", "molar"):
                    raise ValueError("ComponentFlowValue basis must be 'mass' or 'molar'.")
                add_equation(
                    {(stream, ratio.comp_name): flow_conversion(stream, ratio.comp_name, basis)},
                    ratio.target_value,
                    ratio.description,
                )
            else:
                raise TypeError(
                    f"Unsupported ratio constraint type: {type(ratio).__name__}"
                )

        return variable_refs, np.asarray(rows), np.asarray(targets), labels

    def _build_material_balance_dataframe(self) -> pd.DataFrame:
        """Build a table with native, mass-basis, and molar-basis stream results."""
        streams = self.input_streams + self.output_streams
        component_names = sorted(
            {component.name for stream in streams for component in stream.components}
        )
        rows = [
            ("total_flow_rate", ""),
            ("mass_flow_rate", ""),
            ("molar_flow_rate", ""),
        ]
        for variable in (
            "component_flow_rate",
            "fraction",
            "component_mass_flow_rate",
            "component_molar_flow_rate",
            "mass_fraction",
            "mole_fraction",
        ):
            rows.extend((variable, name) for name in component_names)
        table = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(rows, names=("variable", "component"))
        )

        for stream in streams:
            values = {
                ("total_flow_rate", ""): stream.flow_rate,
                ("mass_flow_rate", ""): stream.mass_flow_rate,
                ("molar_flow_rate", ""): stream.molar_flow_rate,
            }
            mass_fractions = stream.mass_fractions
            mole_fractions = stream.mole_fractions
            mass_total = stream.mass_flow_rate
            molar_total = stream.molar_flow_rate
            for component_name in component_names:
                fraction = stream.composition.get(component_name)
                values[("fraction", component_name)] = fraction
                values[("mass_fraction", component_name)] = (
                    None if mass_fractions is None
                    else mass_fractions.get(component_name)
                )
                values[("mole_fraction", component_name)] = (
                    None if mole_fractions is None
                    else mole_fractions.get(component_name)
                )
                values[("component_flow_rate", component_name)] = (
                    None if stream.flow_rate is None or fraction is None
                    else stream.flow_rate * fraction
                    if stream.flow_basis == stream.composition_basis
                    else (
                        mass_total * mass_fractions[component_name]
                        if stream.flow_basis == "mass" and mass_fractions is not None
                        else molar_total * mole_fractions[component_name]
                        if stream.flow_basis == "molar" and mole_fractions is not None
                        else None
                    )
                )
                values[("component_mass_flow_rate", component_name)] = (
                    None if mass_total is None or mass_fractions is None
                    else mass_total * mass_fractions.get(component_name, 0.0)
                )
                values[("component_molar_flow_rate", component_name)] = (
                    None if molar_total is None or mole_fractions is None
                    else molar_total * mole_fractions.get(component_name, 0.0)
                )
            table[stream.name] = pd.Series(values)
        return table

    def solve_material_balances(self) -> pd.DataFrame:
        """Solve balances and stream constraints as a linear feasibility problem.

        Component flow rates on a common mass or molar basis are the decision
        variables. Stream specifications are converted to linear constraints
        using component molar masses where needed. Nonnegative bounds enforce
        physical component flows without an initial guess.

        Returns:
            pd.DataFrame: Complete material balance table.

        Raises:
            ValueError: If the system is under/over-specified, infeasible, or
                does not have a unique solution.
        """
        unknown_refs = self._collect_unknowns()
        if unknown_refs and not self.is_solvable():
            raise ValueError(
                f"Process unit '{self.name}' cannot be solved: "
                f"degrees of freedom are not zero ({self.unknowns} unknowns, "
                f"{self.independent_material_balances} material balances, "
                f"{len(self.ratios)} ratio constraints)."
            )

        streams = self.input_streams + self.output_streams
        component_names = sorted(
            {comp.name for stream in streams for comp in stream.components}
        )
        variable_refs, matrix, targets, labels = self._build_component_flow_system(
            component_names
        )

        result = linprog(
            c=np.zeros(len(variable_refs)),
            A_eq=matrix,
            b_eq=targets,
            bounds=(0.0, None),
            method="highs",
        )
        if not result.success:
            if result.status == 2:
                detail = "constraints are infeasible with nonnegative component flows"
            else:
                detail = result.message
            raise ValueError(
                f"Failed to solve material balances for '{self.name}': {detail}."
            )

        # Reject systems with more than one solution. Normalize equality rows
        # first so the rank check is not affected by different flow magnitudes.
        row_scales = np.maximum(np.max(np.abs(matrix), axis=1), 1.0)
        normalized_matrix = matrix / row_scales[:, np.newaxis]
        rank = np.linalg.matrix_rank(normalized_matrix)
        if rank < len(variable_refs):
            raise ValueError(
                f"Process unit '{self.name}' is underdetermined: constraints "
                "allow more than one component-flow solution."
            )

        values = result.x
        residuals = matrix @ values - targets
        for residual, target, label in zip(residuals, targets, labels):
            if label.startswith("material balance"):
                allowed_error = self.tol
            else:
                allowed_error = self.tol * max(1.0, abs(target))
            if abs(residual) > allowed_error:
                raise ValueError(
                    f"Failed to solve material balances for '{self.name}': "
                    f"{label} residual {residual:.6g} exceeds tolerance "
                    f"{allowed_error:.6g}."
                )

        component_flows = {ref: value for ref, value in zip(variable_refs, values)}

        def flow_in_basis(stream, component_name, basis):
            value = component_flows[(stream, component_name)]
            if self.solver_basis == basis:
                return value
            component = next(c for c in stream.components if c.name == component_name)
            if component.molar_mass is None:
                raise ValueError(
                    f"Molar mass for '{component_name}' is required to convert flow bases."
                )
            return (
                value * component.molar_mass
                if basis == "mass"
                else value / component.molar_mass
            )

        candidate_values = {}
        for stream in streams:
            native_component_flows = {
                name: flow_in_basis(stream, name, stream.flow_basis)
                for name in component_names
            }
            total_flow = sum(native_component_flows.values())
            if total_flow <= self.tol and any(
                fraction is None for fraction in stream.composition.values()
            ):
                raise ValueError(
                    f"Cannot determine composition for zero-flow stream '{stream.name}'."
                )
            composition_flows = {
                name: flow_in_basis(stream, name, stream.composition_basis)
                for name in component_names
            }
            composition_total = sum(composition_flows.values())
            composition = {
                name: (
                    composition_flows[name] / composition_total
                    if composition_total > self.tol
                    else stream.composition[name]
                )
                for name in component_names
            }
            candidate_values[stream] = (total_flow, composition)

        # Cross-multiplied ratio equations need a nonzero original denominator.
        for ratio in self.ratios:
            if isinstance(ratio, FlowRatio):
                denominator_stream = next(
                    stream for stream in streams if stream.name == ratio.stream2_name
                )
                basis = getattr(ratio, "basis", None) or self.solver_basis
                denominator = sum(
                    flow_in_basis(denominator_stream, name, basis)
                    for name in component_names
                )
            elif isinstance(ratio, ComponentFlowRatio):
                denominator_stream = next(
                    stream for stream in streams if stream.name == ratio.stream2_name
                )
                basis = getattr(ratio, "basis", None) or self.solver_basis
                denominator = flow_in_basis(
                    denominator_stream, ratio.comp2_name, basis
                )
            elif isinstance(ratio, CompositionRatio):
                denominator_stream = next(
                    stream for stream in streams if stream.name == ratio.stream_name
                )
                denominator = flow_in_basis(
                    denominator_stream,
                    ratio.comp2_name,
                    denominator_stream.composition_basis,
                )
            else:
                continue
            if denominator <= self.tol:
                raise ValueError(
                    f"Cannot satisfy {ratio.description}: denominator is zero."
                )

        # Commit only after feasibility, uniqueness, and reconstruction checks pass.
        for stream, (total_flow, composition) in candidate_values.items():
            stream.flow_rate = total_flow
            stream.composition.update(composition)

        return self._build_material_balance_dataframe()

    def print_report(self) -> None:
        """Print stream results and material balance residuals in the solver basis."""
        streams = self.input_streams + self.output_streams
        table = self._build_material_balance_dataframe()
        print(f"\nProcess unit report: {self.name}")
        print(f"Material balance basis: {self.solver_basis}")

        for label, grouped_streams in (
            ("Input streams", self.input_streams),
            ("Output streams", self.output_streams),
        ):
            print(f"\n{label}:")
            if not grouped_streams:
                print("  (none)")
                continue
            for stream in grouped_streams:
                flow = (
                    "unknown"
                    if stream.flow_rate is None
                    else f"{stream.flow_rate:.6g} {stream.flow_type}"
                )
                print(f"  {stream.name}: total flow = {flow}")
                for component in stream.components:
                    name = component.name
                    fraction = stream.composition.get(name)
                    fraction_text = "unknown" if fraction is None else f"{fraction:.6g}"
                    component_flow = table.loc[("component_flow_rate", name), stream.name]
                    flow_text = "unknown" if pd.isna(component_flow) else f"{component_flow:.6g}"
                    print(
                        f"    {name}: {stream.composition_basis} fraction = "
                        f"{fraction_text}, component flow = {flow_text} {stream.flow_type}"
                    )
                for basis, total_row in (
                    ("mass", "mass_flow_rate"),
                    ("molar", "molar_flow_rate"),
                ):
                    total = table.loc[(total_row, ""), stream.name]
                    if pd.notna(total):
                        print(f"    derived {basis} flow = {total:.6g}")

        print(f"\n{self.solver_basis.capitalize()} balance checks:")
        for component_name in sorted(
            {component.name for stream in streams for component in stream.components}
        ):
            balance_row = (
                "component_mass_flow_rate"
                if self.solver_basis == "mass"
                else "component_molar_flow_rate"
            )
            input_values = [
                table.loc[(balance_row, component_name), stream.name]
                for stream in self.input_streams
            ]
            output_values = [
                table.loc[(balance_row, component_name), stream.name]
                for stream in self.output_streams
            ]
            input_flow = sum(value for value in input_values if pd.notna(value))
            output_flow = sum(value for value in output_values if pd.notna(value))
            residual = input_flow - output_flow
            status = "PASS" if abs(residual) <= self.tol else "FAIL"
            print(
                f"  {component_name}: input = {input_flow:.6g}, "
                f"output = {output_flow:.6g}, residual = {residual:.6g} [{status}]"
            )
