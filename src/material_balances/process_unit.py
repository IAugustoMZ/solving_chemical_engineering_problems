"""
ProcessUnit class for modeling process units in material balance problems.

This module provides the ProcessUnit class for setting up and solving
material balance problems around process units with inlet and outlet streams.
"""

import numpy as np
from scipy.optimize import least_squares


class ProcessUnit:
    """
    Represents a process unit with inlet and outlet streams.

    A ProcessUnit encapsulates a single piece of equipment or process step
    (e.g., evaporator, mixer, reactor) and performs material balance calculations
    across it. It validates stream consistency, performs degree-of-freedom analysis,
    and solves for unknown flow rates and compositions.

    The number of independent material balance equations equals the number of
    components. The system is solvable when unknowns <= independent material balances.

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
        """
        Check if the system has sufficient equations to solve for unknowns.

        A system is solvable when:
        degrees_of_freedom = unknowns - independent_material_balances - len(ratios) <= 0

        Each ratio constraint reduces degrees of freedom by 1.

        Returns:
            bool: True if solvable, False otherwise.

        Example:
            >>> if evap.is_solvable():
            ...     evap.solve_material_balances()
        """
        dof = self.unknowns - self.independent_material_balances - len(self.ratios)
        return dof <= 0

    def report_degrees_of_freedom(self) -> None:
        """
        Print a detailed degrees of freedom (DOF) analysis.

        Shows the number of unknowns, independent equations, constraints,
        and resulting degrees of freedom to help users understand why the
        system is solvable or not.
        """
        dof = self.unknowns - self.independent_material_balances - len(self.ratios)

        print(f"\nDegrees of Freedom Analysis for '{self.name}':")
        print(f"  Unknowns (flow rates + compositions): {self.unknowns}")
        print(f"  Independent material balance equations: {self.independent_material_balances}")
        print(f"  Ratio constraints: {len(self.ratios)}")
        print(f"  Degrees of freedom: {dof}")

        if dof <= 0:
            print(f"  Status: SOLVABLE (need to provide {-dof} more value{'s' if dof != -1 else ''})")
        else:
            print(f"  Status: UNDERDETERMINED (need to provide {dof} more value{'s' if dof != 1 else ''})")

    def suggest_missing_information(self) -> None:
        """
        Print suggestions for missing information if the system is not solvable.

        If the system is underdetermined (more unknowns than equations),
        calculates how many more values are needed and lists candidates.
        """
        if self.is_solvable():
            print(
                f"Process unit '{self.name}' is already solvable. "
                "No additional information needed."
            )
            return

        dof = self.unknowns - self.independent_material_balances - len(self.ratios)
        print(
            f"To make the system solvable, consider providing {dof} more "
            f"value{'s' if dof != 1 else ''}. Options:"
        )

        missing_info = []

        for stream in self.input_streams + self.output_streams:
            if stream.flow_rate is None:
                missing_info.append(f"Stream '{stream.name}': flow rate")

            for comp_name, comp_value in stream.composition.items():
                if comp_value is None:
                    missing_info.append(
                        f"Stream '{stream.name}': composition for '{comp_name}'"
                    )

        for info in missing_info:
            print(f"  - {info}")

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

    def _apply_unknowns(self, unknown_refs: list, values: list) -> None:
        """
        Apply solved values back to streams.

        Updates stream flow_rate and composition values from the solver result.

        Parameters:
            unknown_refs (list): List of (stream, kind, key) tuples from _collect_unknowns.
            values (list): List of solved values from optimizer.
        """
        for (stream, kind, key), value in zip(unknown_refs, values):
            if kind == "flow_rate":
                stream.flow_rate = value
            else:
                stream.composition[key] = value

    def _material_balance_residuals(
        self, values: list, unknown_refs: list, component_names: list
    ) -> list:
        """
        Calculate residuals for material balance and ratio constraint equations.

        For each component, residual = sum(input flows) - sum(output flows).
        For each ratio constraint, residual = actual_ratio - target_ratio.
        At the solution, all residuals should be ~0.

        Parameters:
            values (list): Current unknown values from optimizer.
            unknown_refs (list): List of (stream, kind, key) tuples.
            component_names (list): Sorted list of unique component names.

        Returns:
            list: Residual values for each component balance and ratio constraint.
        """
        self._apply_unknowns(unknown_refs, values)

        residuals = []

        # Material balance residuals
        for comp_name in component_names:
            input_total = sum(
                stream.flow_rate * stream.composition[comp_name]
                for stream in self.input_streams
            )
            output_total = sum(
                stream.flow_rate * stream.composition[comp_name]
                for stream in self.output_streams
            )
            residuals.append(input_total - output_total)

        # Ratio constraint residuals
        streams_dict = {
            stream.name: stream for stream in self.input_streams + self.output_streams
        }
        for ratio in self.ratios:
            residuals.append(ratio.compute_residual(streams_dict))

        return residuals

    def solve_material_balances(self) -> np.ndarray:
        """
        Solve the material balance system for unknown flow rates and compositions.

        Uses scipy's least_squares optimizer to minimize the sum of squared
        residuals from material balance equations. All flow rates and compositions
        are constrained to be non-negative.

        Returns:
            np.ndarray: Array of solved values for unknowns, or None if unsolvable.

        Raises:
            ValueError: If the system is not solvable.

        Example:
            >>> evap.solve_material_balances()
            >>> print(f"Vapor flow: {evap.output_streams[0].flow_rate}")
        """
        if not self.is_solvable():
            raise ValueError(
                f"Process unit '{self.name}' cannot be solved: "
                f"{self.unknowns} unknowns > {self.independent_material_balances} "
                f"independent material balances"
            )

        unknown_refs = self._collect_unknowns()

        if not unknown_refs:
            return None

        component_names = sorted(
            {
                comp.name
                for stream in self.input_streams + self.output_streams
                for comp in stream.components
            }
        )

        known_input_flow = (
            sum(
                stream.flow_rate
                for stream in self.input_streams
                if stream.flow_rate is not None
            )
            or 1.0
        )
        x0 = [
            known_input_flow if kind == "flow_rate" else 1.0 / len(stream.components)
            for stream, kind, _ in unknown_refs
        ]

        result = least_squares(
            self._material_balance_residuals,
            x0=x0,
            args=(unknown_refs, component_names),
            bounds=(0.0, np.inf),
        )

        if not result.success:
            raise ValueError(
                f"Failed to solve material balances for '{self.name}': "
                f"{result.message}"
            )

        self._apply_unknowns(unknown_refs, result.x)
        return result.x

    def print_report(self) -> None:
        """
        Print a detailed report of streams and material balance checks.

        Shows flow rates and compositions for all streams, and calculates
        residuals (input - output) for each component to verify the balance.
        """
        streams = self.input_streams + self.output_streams
        component_names = sorted(
            {comp.name for stream in streams for comp in stream.components}
        )

        print(f"\nProcess unit report: {self.name}")
        print(f"Mass balance tolerance: {self.tol:.1e}")

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
                for component_name in component_names:
                    fraction = stream.composition.get(component_name)
                    fraction_text = "unknown" if fraction is None else f"{fraction:.6g}"
                    if stream.flow_rate is None or fraction is None:
                        component_flow_text = "unknown"
                    else:
                        component_flow_text = (
                            f"{stream.flow_rate * fraction:.6g} {stream.flow_type}"
                        )
                    print(
                        f"    {component_name}: fraction = {fraction_text}, "
                        f"component flow = {component_flow_text}"
                    )

        print("\nMass balance checks:")
        for component_name in component_names:
            input_flow = (
                sum(
                    stream.flow_rate * stream.composition[component_name]
                    for stream in self.input_streams
                )
                if all(
                    stream.flow_rate is not None
                    and stream.composition.get(component_name) is not None
                    for stream in self.input_streams
                )
                else None
            )
            output_flow = (
                sum(
                    stream.flow_rate * stream.composition[component_name]
                    for stream in self.output_streams
                )
                if all(
                    stream.flow_rate is not None
                    and stream.composition.get(component_name) is not None
                    for stream in self.output_streams
                )
                else None
            )

            if input_flow is None or output_flow is None:
                print(
                    f"  {component_name}: unavailable "
                    "(stream flow or composition is unknown)"
                )
                continue

            residual = input_flow - output_flow
            status = "PASS" if abs(residual) <= self.tol else "FAIL"
            print(
                f"  {component_name}: input = {input_flow:.6g}, "
                f"output = {output_flow:.6g}, residual = {residual:.6g} [{status}]"
            )

        if any(stream.flow_rate is None for stream in streams):
            print("  Overall: unavailable (one or more stream flow rates are unknown)")
        else:
            total_input = sum(stream.flow_rate for stream in self.input_streams)
            total_output = sum(stream.flow_rate for stream in self.output_streams)
            residual = total_input - total_output
            status = "PASS" if abs(residual) <= self.tol else "FAIL"
            print(
                f"  Overall: input = {total_input:.6g}, "
                f"output = {total_output:.6g}, residual = {residual:.6g} [{status}]"
            )
