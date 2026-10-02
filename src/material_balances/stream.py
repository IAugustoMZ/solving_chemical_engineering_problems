"""
Stream and Component classes for representing process streams in material balance problems.

This module provides the Component class (representing chemical species) and the Stream class
(representing material flows with composition), along with StreamFactory for convenient
multi-stream creation from a shared component list.
"""

from collections.abc import Mapping

import numpy as np


class Component:
    """
    Represents a chemical component or species in a process stream.

    A Component is a fundamental entity in material balance calculations,
    representing individual chemical species (e.g., H2O, C2H5OH, C12H22O11).
    Components are referenced by name and used in composition calculations
    for streams and material balances for process units.

    Attributes:
        name (str): The name or identifier of the component (e.g., "Water",
                    "Acetone", "Sugar").
        molar_mass (float or None): Positive mass per mole value used for
                                    mass/molar conversions.

    Example:
        >>> water = Component("Water")
        >>> water.name
        'Water'
    """

    def __init__(self, name: str, molar_mass: float = None) -> None:
        """
        Initialize a Component with a given name.

        Parameters:
            name (str): The name or identifier of the chemical component.

        Example:
            >>> acetone = Component("Acetone")
        """
        if molar_mass is not None and (not np.isfinite(molar_mass) or molar_mass <= 0):
            raise ValueError("molar_mass must be a finite positive value.")
        self.name = name
        self.molar_mass = molar_mass


def _basis_from_type(flow_type: str) -> str:
    """Map a flow unit label to its mass or molar basis."""
    normalized = str(flow_type).strip().lower()
    if any(token in normalized for token in ("mole", "molar", "mol", "kmol")):
        return "molar"
    if any(token in normalized for token in ("mass", "kg", "gram", " g", "ton")):
        return "mass"
    if "volume" in normalized or normalized in ("volumetric", "m3/h", "l/min"):
        return "volume"
    raise ValueError(
        f"Unsupported flow_type '{flow_type}': use mass, molar, or legacy volumetric units."
    )


class Stream:
    """
    Represents a process stream with flow rate and composition.

    A Stream is a fundamental entity in material balance calculations, representing
    a flow of material at a specific point in a process. Each stream has a flow rate,
    composition (mass or mole fractions), and list of components.

    The Stream class automatically validates composition data and can calculate
    missing composition values when all but one component's fraction is specified.

    Attributes:
        name (str): Identifier for the stream (e.g., "Feed", "Product").
        flow_rate (float or None): Volumetric, mass, or molar flow rate. Can be None
                                   if unknown and to be solved for.
        flow_type (str): Type of flow rate (e.g., "mole", "mass", "volume").
        components (list): List of Component objects present in the stream.
        composition (dict[str, float or None]): Mole/mass fractions for each component.
                                                Can contain one None value for unknown.
        direction (str): Direction of flow ("input" or "output").
    """

    def __init__(
        self,
        name: str,
        flow_rate: float,
        flow_type: str,
        components: list,
        composition: dict,
        direction: str = "input",
        composition_basis: str = None,
    ) -> None:
        """
        Initialize a Stream with flow rate and composition data.

        Validates that the number of components matches composition values and
        that specified compositions sum to <= 1.0. Multiple composition values
        can be None and will be solved for if the system is properly determined.

        Parameters:
            name (str): Stream identifier or tag.
            flow_rate (float): Flow rate value. Can be None if unknown.
            flow_type (str): Type of flow rate (e.g., "mole", "mass").
            components (list): List of Component objects present in the stream.
            composition (dict): Dictionary mapping component names to fractions.
                               Can have multiple None values for unknown compositions.
            direction (str, optional): Flow direction, "input" or "output".
                                      Defaults to "input".
            composition_basis (str, optional): "mass" or "molar"; defaults to the
                                              selected flow basis.

        Raises:
            ValueError: If number of components doesn't match composition values,
                       or if specified composition values don't sum to <= 1.0.

        Example:
            >>> water = Component("Water")
            >>> acetone = Component("Acetone")
            >>> stream = Stream(
            ...     name="evaporator_feed",
            ...     flow_rate=10.0,
            ...     flow_type="mole",
            ...     components=[water, acetone],
            ...     composition={"Water": 0.65, "Acetone": None}
            ... )
            >>> stream.composition["Acetone"]  # Will be solved for, not auto-calculated
            None
        """
        if flow_rate is not None and flow_rate < 0:
            raise ValueError("flow_rate must be nonnegative.")
        self.name = name
        self.flow_rate = flow_rate
        self.flow_type = flow_type
        self.flow_basis = _basis_from_type(flow_type)
        self.composition_basis = composition_basis or self.flow_basis
        if self.composition_basis not in ("mass", "molar", "volume"):
            raise ValueError("composition_basis must be 'mass' or 'molar'.")
        if self.composition_basis == "volume" and self.flow_basis != "volume":
            raise ValueError(
                "Volume composition cannot be used with mass or molar flow."
            )
        self.components = components
        self.composition = composition
        self.direction = direction

        self._validate_components_composition()
        self._validate_composition_sum()

    def _molar_flows_from_known_data(self):
        """Return component molar flows if total flow and composition are known."""
        if (
            self.flow_rate is None
            or self.flow_basis == "volume"
            or any(v is None for v in self.composition.values())
        ):
            return None
        fractions = {c.name: self.composition[c.name] for c in self.components}
        if self.flow_basis == self.composition_basis == "molar":
            return {
                name: self.flow_rate * fraction for name, fraction in fractions.items()
            }

        # A component whose specified fraction is zero has zero flow on either
        # mass or molar basis. Its molar mass is therefore not needed for a
        # conversion. This matters for pure tracer streams represented with a
        # shared component list (for example, pure CO2 plus a carrier gas of
        # unknown molar mass).
        missing_active_molar_mass = any(
            component.molar_mass is None and fractions[component.name] != 0
            for component in self.components
        )
        if missing_active_molar_mass:
            return None
        if self.composition_basis == "molar":
            average_mass = sum(
                fractions[component.name] * component.molar_mass
                for component in self.components
                if fractions[component.name] != 0
            )
            total_moles = (
                self.flow_rate
                if self.flow_basis == "molar"
                else self.flow_rate / average_mass
            )
            return {
                name: total_moles * fraction for name, fraction in fractions.items()
            }
        mass_per_mole = sum(
            fractions[component.name] / component.molar_mass
            for component in self.components
            if fractions[component.name] != 0
        )
        total_mass = (
            self.flow_rate
            if self.flow_basis == "mass"
            else self.flow_rate / mass_per_mole
        )
        return {
            component.name: (
                0.0
                if fractions[component.name] == 0
                else total_mass * fractions[component.name] / component.molar_mass
            )
            for component in self.components
        }

    @property
    def molar_flow_rate(self):
        """Molar flow rate, or its derived value when the stream is mass-based."""
        if self.flow_rate is not None and self.flow_basis == "molar":
            return self.flow_rate
        flows = self._molar_flows_from_known_data()
        return None if flows is None else sum(flows.values())

    @property
    def mass_flow_rate(self):
        """Mass flow rate, or its derived value when the stream is molar-based."""
        if self.flow_rate is not None and self.flow_basis == "mass":
            return self.flow_rate
        flows = self._molar_flows_from_known_data()
        if flows is None or any(
            component.molar_mass is None and flows[component.name] != 0
            for component in self.components
        ):
            return None
        return sum(
            (
                0.0
                if flows[component.name] == 0
                else flows[component.name] * component.molar_mass
            )
            for component in self.components
        )

    @property
    def mole_fractions(self):
        """Derived mole fractions, or None until sufficient data are known."""
        if self.composition_basis == "molar" and all(
            v is not None for v in self.composition.values()
        ):
            return dict(self.composition)
        flows = self._molar_flows_from_known_data()
        if flows is None:
            return None
        total = sum(flows.values())
        return {name: value / total for name, value in flows.items()} if total else None

    @property
    def mass_fractions(self):
        """Derived mass fractions, or None until sufficient data are known."""
        if self.composition_basis == "mass" and all(
            v is not None for v in self.composition.values()
        ):
            return dict(self.composition)
        flows = self._molar_flows_from_known_data()
        if flows is None or any(
            component.molar_mass is None and flows[component.name] != 0
            for component in self.components
        ):
            return None
        masses = {
            component.name: (
                0.0
                if flows[component.name] == 0
                else flows[component.name] * component.molar_mass
            )
            for component in self.components
        }
        total = sum(masses.values())
        return (
            {name: value / total for name, value in masses.items()} if total else None
        )

    def _validate_components_composition(self) -> None:
        """
        Validate that number of components matches number of composition values.

        Raises:
            ValueError: If number of components doesn't match number of composition values.
        """
        component_names = [component.name for component in self.components]
        if len(component_names) != len(set(component_names)):
            raise ValueError("Component names in a stream must be unique.")
        if len(component_names) != len(self.composition):
            raise ValueError(
                "Number of components must match the number of composition values."
            )
        if set(component_names) != set(self.composition):
            raise ValueError(
                "Composition keys must match the component names in the stream."
            )

    def _validate_composition_sum(self) -> None:
        """
        Validate that specified composition values sum to <= 1.0.

        If all values are specified, they must sum to exactly 1.0.
        If some values are None, specified values must sum to <= 1.0
        (the missing values will be solved for).

        Raises:
            ValueError: If specified compositions exceed 1.0 or if all
                       are specified but don't sum to 1.0.
        """
        specified_values = [v for v in self.composition.values() if v is not None]
        if not specified_values:
            return

        if any(value < 0 or value > 1 for value in specified_values):
            raise ValueError("Composition fractions must be between 0 and 1.")
        specified_sum = sum(specified_values)

        if all(value is not None for value in self.composition.values()):
            if not np.isclose(specified_sum, 1.0):
                raise ValueError("Composition values must sum to 1.")
        else:
            if specified_sum > 1.0:
                raise ValueError(
                    "Specified composition values must not exceed 1.0 "
                    "(missing values must sum to a non-negative amount)."
                )

            # Auto-calculate single missing composition as 1 - sum(known)
            none_count = list(self.composition.values()).count(None)
            if none_count == 1:
                for key, value in self.composition.items():
                    if value is None:
                        self.composition[key] = 1.0 - specified_sum
                        break


class StreamFactory:
    """
    Factory for creating and managing multiple streams with a shared component list.

    StreamFactory simplifies the creation of multiple streams by requiring a component
    list to be specified once. Streams are created via add_stream() using positional
    compositions (in the same order as the factory's component list) rather than
    manually constructing composition dicts. The factory also manages ratio constraints,
    validating that ratios only reference registered streams and components.

    Attributes:
        component_names (list[str]): List of component names (e.g., ["Water", "Acetone"]).
        components (list[Component]): Corresponding Component objects (shared across all streams).
        default_flow_type (str or None): Default mass- or molar-flow unit applied
                                        unless overridden per add_stream call.
        molar_masses (dict[str, float] or list[float | None]): Optional molar
            masses, supplied either as a mapping by component name or as a list
            aligned with ``component_names``.
        streams (dict[str, Stream]): Registry of created streams, keyed by name.
        ratios (list[Ratio]): List of registered ratio constraints (validated against streams).

    Example:
        >>> factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
        >>> feed = factory.add_stream("Feed", [0.65, 0.35], flow_rate=10)
        >>> vapor = factory.add_stream("Vapor", [0.25, None], direction="output")
        >>> from src.material_balances import FlowRatio
        >>> ratio = FlowRatio("Feed", "Vapor", target_ratio=3.45)
        >>> factory.add_ratio(ratio)
        >>> unit = factory.build_process_unit("Evaporator")
    """

    def __init__(
        self,
        component_names: list,
        default_flow_type: str = None,
        molar_masses: dict | list | tuple = None,
    ) -> None:
        """
        Initialize a StreamFactory with a shared component list.

        Parameters:
            component_names (list[str]): Names of all components present in streams
                                        created by this factory (e.g., ["Water", "Acetone"]).
            default_flow_type (str, optional): Default mass- or molar-flow unit.
            molar_masses (dict or list, optional): Molar masses supplied either
                as a mapping of component names to values or positionally in
                the same order as ``component_names``. Use ``None`` for a
                component whose molar mass is intentionally unavailable.

        Example:
            >>> factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
        """
        self.component_names = component_names
        if molar_masses is None:
            molar_mass_by_name = {}
        elif isinstance(molar_masses, Mapping):
            molar_mass_by_name = dict(molar_masses)
        elif isinstance(molar_masses, (list, tuple)):
            if len(molar_masses) != len(component_names):
                raise ValueError(
                    "Molar-mass length "
                    f"({len(molar_masses)}) must match component count "
                    f"({len(component_names)})."
                )
            molar_mass_by_name = dict(zip(component_names, molar_masses))
        else:
            raise TypeError(
                "molar_masses must be a mapping or a list aligned with component_names."
            )

        unknown_molar_mass_names = set(molar_mass_by_name) - set(component_names)
        if unknown_molar_mass_names:
            raise ValueError(
                "Molar masses were provided for unknown components: "
                + ", ".join(sorted(map(str, unknown_molar_mass_names)))
            )
        self.components = [
            Component(name, molar_mass_by_name.get(name)) for name in component_names
        ]
        self.default_flow_type = default_flow_type
        self.streams: dict[str, Stream] = {}
        self.ratios: list = []

    def add_stream(
        self,
        name: str,
        compositions: list,
        flow_rate: float = None,
        flow_type: str = None,
        direction: str = "input",
        composition_basis: str = None,
    ) -> Stream:
        """
        Create and register a stream using positional compositions.

        Parameters:
            name (str): Unique identifier for the stream (e.g., "Feed", "Product").
            compositions (list[float or None]): Component fractions in the same order
                                               as the factory's component_names.
                                               Can include one None for auto-calculation.
            flow_rate (float, optional): Flow rate value. None if unknown (to be solved).
                                        Defaults to None.
            flow_type (str, optional): Type of flow rate (e.g., "mole", "mass", "volume").
                                      If not provided, uses factory's default_flow_type.
                                      Defaults to None.
            direction (str, optional): Flow direction, "input" or "output".
                                      Defaults to "input".

        Returns:
            Stream: The newly created and registered Stream object.

        Raises:
            ValueError: If stream name is already registered, composition length doesn't
                       match component count, or if flow_type is required but missing.

        Example:
            >>> factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
            >>> feed = factory.add_stream("Feed", [0.65, 0.35], flow_rate=10)
            >>> vapor = factory.add_stream("Vapor", [0.25, None], direction="output")
        """
        if name in self.streams:
            raise ValueError(f"Stream '{name}' is already registered in this factory.")

        if len(compositions) != len(self.components):
            raise ValueError(
                f"Composition length ({len(compositions)}) must match component count "
                f"({len(self.components)})."
            )

        resolved_flow_type = flow_type or self.default_flow_type
        if resolved_flow_type is None:
            raise ValueError(
                "flow_type must be provided either per stream or via factory default."
            )

        composition = dict(zip(self.component_names, compositions))

        stream = Stream(
            name=name,
            flow_rate=flow_rate,
            flow_type=resolved_flow_type,
            components=self.components,
            composition=composition,
            direction=direction,
            composition_basis=composition_basis,
        )

        self.streams[name] = stream
        return stream

    def get_stream(self, name: str) -> Stream:
        """
        Retrieve a registered stream by name.

        Parameters:
            name (str): The stream identifier.

        Returns:
            Stream: The registered Stream object.

        Raises:
            KeyError: If the stream name is not registered.

        Example:
            >>> factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
            >>> feed = factory.add_stream("Feed", [0.65, 0.35], flow_rate=10)
            >>> same_feed = factory.get_stream("Feed")
        """
        return self.streams[name]

    def add_ratio(self, ratio_type="flow", **kwargs):
        """
        Create and register a ratio constraint from raw parameters.

        The factory automatically instantiates the appropriate Ratio subclass
        (FlowRatio, CompositionRatio, ComponentFlowRatio, or ComponentFlowValue)
        based on ratio_type and validates that all referenced streams and
        components are registered.

        Parameters:
            ratio_type (str): Type of ratio constraint. Options:
                - "flow": FlowRatio between two stream flow rates
                - "composition": CompositionRatio between two component compositions
                - "component_flow": ComponentFlowRatio between component flows
                - "component_flow_value": ComponentFlowValue for known absolute component flow

                Defaults to "flow".

            **kwargs: Arguments specific to the ratio_type:

                For "flow": stream1, stream2, target_ratio
                    >>> factory.add_ratio("flow", stream1="Feed", stream2="Vapor", target_ratio=3.45)

                For "composition": stream, comp1, comp2, target_ratio
                    >>> factory.add_ratio("composition", stream="Outlet", comp1="Acetone",
                    ...                  comp2="Water", target_ratio=4.0)

                For "component_flow": stream1, comp1, stream2, comp2, target_ratio
                    >>> factory.add_ratio("component_flow", stream1="Strawberry", comp1="Solids",
                    ...                  stream2="Sugar", comp2="Sugar", target_ratio=0.45)

                For "component_flow_value": stream, comp, target_value
                    >>> factory.add_ratio("component_flow_value", stream="Product", comp="Solids",
                    ...                  target_value=500.0)

        Returns:
            Ratio: The newly created and registered Ratio object.

        Raises:
            ValueError: If ratio_type is unknown, required parameters are missing,
                       or if the ratio references a stream or component not in this factory.

        Example:
            >>> factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
            >>> feed = factory.add_stream("Feed", [0.65, 0.35], flow_rate=10)
            >>> vapor = factory.add_stream("Vapor", [0.25, None], direction="output")
            >>> factory.add_ratio("flow", stream1="Feed", stream2="Vapor", target_ratio=3.45)
        """
        from .ratio_constraints import (
            ComponentFlowRatio,
            ComponentFlowValue,
            CompositionRatio,
            FlowRatio,
        )

        if ratio_type == "flow":
            ratio = FlowRatio(
                kwargs["stream1"],
                kwargs["stream2"],
                kwargs["target_ratio"],
                basis=kwargs.get("basis"),
            )
        elif ratio_type == "composition":
            ratio = CompositionRatio(
                kwargs["stream"],
                kwargs["comp1"],
                kwargs["comp2"],
                kwargs["target_ratio"],
            )
        elif ratio_type == "component_flow":
            ratio = ComponentFlowRatio(
                kwargs["stream1"],
                kwargs["comp1"],
                kwargs["stream2"],
                kwargs["comp2"],
                kwargs["target_ratio"],
                basis=kwargs.get("basis"),
            )
        elif ratio_type == "component_flow_value":
            ratio = ComponentFlowValue(
                kwargs["stream"],
                kwargs["comp"],
                kwargs["target_value"],
                basis=kwargs.get("basis"),
            )
        else:
            raise ValueError(
                f"Unknown ratio_type '{ratio_type}'. "
                f"Choose from: 'flow', 'composition', 'component_flow', 'component_flow_value'."
            )

        try:
            ratio.validate_references(self.streams)
        except (KeyError, ValueError) as e:
            raise ValueError(f"Ratio ({ratio.description}) has invalid references: {e}")

        self.ratios.append(ratio)
        return ratio

    def build_process_unit(self, name: str):
        """
        Build a ProcessUnit from registered streams and ratios.

        Splits registered streams into input and output groups based on direction,
        then constructs a ProcessUnit with all registered ratios.

        Parameters:
            name (str): Unit identifier or tag for the ProcessUnit.

        Returns:
            ProcessUnit: A ProcessUnit ready to solve (or to verify solvability).

        Example:
            >>> factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
            >>> feed = factory.add_stream("Feed", [0.65, 0.35], flow_rate=10)
            >>> vapor = factory.add_stream("Vapor", [0.25, None], direction="output")
            >>> liquid = factory.add_stream("Liquid", [0.187, None], direction="output")
            >>> unit = factory.build_process_unit("Evaporator")
            >>> unit.solve_material_balances()
        """
        from .process_unit import ProcessUnit

        input_streams = [s for s in self.streams.values() if s.direction == "input"]
        output_streams = [s for s in self.streams.values() if s.direction == "output"]

        return ProcessUnit(
            name=name,
            input_streams=input_streams,
            output_streams=output_streams,
            ratios=self.ratios,
        )
