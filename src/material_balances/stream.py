"""
Stream and Component classes for representing process streams in material balance problems.

This module provides the Component class (representing chemical species) and the Stream class
(representing material flows with composition), along with StreamFactory for convenient
multi-stream creation from a shared component list.
"""

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

    Example:
        >>> water = Component("Water")
        >>> water.name
        'Water'
    """

    def __init__(self, name: str) -> None:
        """
        Initialize a Component with a given name.

        Parameters:
            name (str): The name or identifier of the chemical component.

        Example:
            >>> acetone = Component("Acetone")
        """
        self.name = name


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
    ) -> None:
        """
        Initialize a Stream with flow rate and composition data.

        Validates that the number of components matches composition values and
        that compositions sum to 1.0 (if all specified). Automatically calculates
        the complementary composition value if exactly one is None.

        Parameters:
            name (str): Stream identifier or tag.
            flow_rate (float): Flow rate value. Can be None if unknown.
            flow_type (str): Type of flow rate (e.g., "mole", "mass").
            components (list): List of Component objects present in the stream.
            composition (dict): Dictionary mapping component names to fractions.
                               Can have one None value for unknown composition.
            direction (str, optional): Flow direction, "input" or "output".
                                      Defaults to "input".

        Raises:
            ValueError: If number of components doesn't match composition values,
                       or if composition values don't sum to ~1.0 when all specified.

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
            >>> stream.composition["Acetone"]
            0.35
        """
        self.name = name
        self.flow_rate = flow_rate
        self.flow_type = flow_type
        self.components = components
        self.composition = composition
        self.direction = direction

        self._validate_components_composition()
        self._validate_composition_sum()
        self.calculate_complementary_composition()

    def _validate_components_composition(self) -> None:
        """
        Validate that number of components matches number of composition values.

        Raises:
            ValueError: If number of components doesn't match number of composition values.
        """
        if len(self.components) != len(self.composition):
            raise ValueError(
                "Number of components must match the number of composition values."
            )

    def _validate_composition_sum(self) -> None:
        """
        Validate that specified composition values sum to 1.0.

        Only validates if all composition values are specified (no None values).

        Raises:
            ValueError: If specified compositions don't sum to 1.0.
        """
        if all(value is not None for value in self.composition.values()):
            if not np.isclose(sum(self.composition.values()), 1.0):
                raise ValueError("Composition values must sum to 1.")

    def calculate_complementary_composition(self) -> None:
        """
        Calculate missing composition value if exactly one is None.

        If exactly one component's composition is None, calculate it as
        1.0 minus the sum of all other known composition values.

        Raises:
            ValueError: If more than one composition value is missing.
        """
        none_count = sum(1 for v in self.composition.values() if v is None)

        if none_count > 1:
            raise ValueError(
                "Cannot calculate complementary composition: "
                "more than one composition value is missing."
            )

        if none_count == 1:
            known_sum = sum(v for v in self.composition.values() if v is not None)
            missing_component = next(
                key for key, value in self.composition.items() if value is None
            )
            self.composition[missing_component] = 1.0 - known_sum


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
        default_flow_type (str or None): Default flow type (e.g., "mole", "mass")
                                        applied if not overridden per add_stream call.
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
        self, component_names: list, default_flow_type: str = None
    ) -> None:
        """
        Initialize a StreamFactory with a shared component list.

        Parameters:
            component_names (list[str]): Names of all components present in streams
                                        created by this factory (e.g., ["Water", "Acetone"]).
            default_flow_type (str, optional): Default flow type (e.g., "mole", "mass").
                                             If set, add_stream calls can omit flow_type.
                                             If None, flow_type is required on each add_stream.
                                             Defaults to None.

        Example:
            >>> factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
        """
        self.component_names = component_names
        self.components = [Component(name) for name in component_names]
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
            raise ValueError(
                f"Stream '{name}' is already registered in this factory."
            )

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
        (FlowRatio, CompositionRatio, or ComponentFlowRatio) based on ratio_type
        and validates that all referenced streams and components are registered.

        Parameters:
            ratio_type (str): Type of ratio constraint. Options:
                - "flow": FlowRatio between two stream flow rates
                - "composition": CompositionRatio between two component compositions
                - "component_flow": ComponentFlowRatio between component flows

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
        from .ratio_constraints import ComponentFlowRatio, CompositionRatio, FlowRatio

        if ratio_type == "flow":
            ratio = FlowRatio(
                kwargs["stream1"], kwargs["stream2"], kwargs["target_ratio"]
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
            )
        else:
            raise ValueError(
                f"Unknown ratio_type '{ratio_type}'. "
                f"Choose from: 'flow', 'composition', 'component_flow'."
            )

        try:
            ratio.validate_references(self.streams)
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"Ratio ({ratio.description}) has invalid references: {e}"
            )

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

        input_streams = [
            s for s in self.streams.values() if s.direction == "input"
        ]
        output_streams = [
            s for s in self.streams.values() if s.direction == "output"
        ]

        return ProcessUnit(
            name=name,
            input_streams=input_streams,
            output_streams=output_streams,
            ratios=self.ratios,
        )
