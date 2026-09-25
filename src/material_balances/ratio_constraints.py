"""
Ratio constraint classes for process unit modeling.

This module provides ratio constraint classes that represent independent
relationships between streams or components (e.g., flow rate ratios,
component flow ratios, composition ratios). These constraints integrate
with ProcessUnit to reduce degrees of freedom in material balance problems.
"""

from abc import ABC, abstractmethod


class Ratio(ABC):
    """
    Abstract base class for ratio constraints.

    A ratio constraint represents an independent algebraic relationship
    between streams or components that reduces the system's degrees of
    freedom by 1 for each independent constraint applied.
    """

    @abstractmethod
    def compute_residual(self, streams_dict):
        """
        Compute the residual (actual_ratio - target_ratio).

        At the solution, this residual should be ~0. The residual is used
        as an objective function term in the least_squares solver.

        Parameters:
            streams_dict (dict): Mapping of stream names to Stream objects.

        Returns:
            float: The residual value (actual_ratio - target_ratio).
        """

    @abstractmethod
    def validate_references(self, streams_dict):
        """
        Validate that all referenced streams and components exist.

        Parameters:
            streams_dict (dict): Mapping of stream names to Stream objects.

        Raises:
            KeyError: If a referenced stream or component doesn't exist.
            ValueError: If the constraint references are invalid.
        """

    @property
    @abstractmethod
    def description(self):
        """
        Return a human-readable description of the constraint.

        Returns:
            str: Description suitable for error messages and logging.
        """


class FlowRatio(Ratio):
    """
    Represents a ratio between two stream flow rates.

    Constraint: stream1.flow_rate / stream2.flow_rate = target_ratio

    Example:
        >>> ratio = FlowRatio("Strawberry", "Sugar", target_ratio=45/55)
        >>> # Constraint: F_strawberry / F_sugar = 45/55
    """

    def __init__(self, stream1_name, stream2_name, target_ratio):
        """
        Initialize a flow rate ratio constraint.

        Parameters:
            stream1_name (str): Name of the first stream (numerator).
            stream2_name (str): Name of the second stream (denominator).
            target_ratio (float): Target ratio value.
        """
        self.stream1_name = stream1_name
        self.stream2_name = stream2_name
        self.target_ratio = target_ratio

    def compute_residual(self, streams_dict):
        """Compute (F1 / F2) - target_ratio."""
        s1 = streams_dict[self.stream1_name]
        s2 = streams_dict[self.stream2_name]

        if s2.flow_rate == 0:
            return float("inf")

        actual_ratio = s1.flow_rate / s2.flow_rate
        return actual_ratio - self.target_ratio

    def validate_references(self, streams_dict):
        """Verify both streams exist."""
        if self.stream1_name not in streams_dict:
            raise KeyError(f"Stream '{self.stream1_name}' not found in process unit")
        if self.stream2_name not in streams_dict:
            raise KeyError(f"Stream '{self.stream2_name}' not found in process unit")

    @property
    def description(self):
        """Return constraint description."""
        return (
            f"FlowRatio: {self.stream1_name}.F / {self.stream2_name}.F = "
            f"{self.target_ratio}"
        )


class ComponentFlowRatio(Ratio):
    """
    Represents a ratio between component flow rates in two streams.

    Constraint: (stream1.flow_rate × stream1.composition[comp1]) /
                (stream2.flow_rate × stream2.composition[comp2]) = target_ratio

    Example:
        >>> ratio = ComponentFlowRatio(
        ...     "Strawberry", "Solids",
        ...     "Sugar", "Sugar",
        ...     target_ratio=0.45
        ... )
        >>> # Constraint: (F_straw × x_solids) / (F_sugar × x_sugar) = 0.45
    """

    def __init__(
        self, stream1_name, comp1_name, stream2_name, comp2_name, target_ratio
    ):
        """
        Initialize a component flow rate ratio constraint.

        Parameters:
            stream1_name (str): Name of the first stream.
            comp1_name (str): Name of component in first stream.
            stream2_name (str): Name of the second stream.
            comp2_name (str): Name of component in second stream.
            target_ratio (float): Target ratio value.
        """
        self.stream1_name = stream1_name
        self.comp1_name = comp1_name
        self.stream2_name = stream2_name
        self.comp2_name = comp2_name
        self.target_ratio = target_ratio

    def compute_residual(self, streams_dict):
        """Compute (F1 × x1_comp1) / (F2 × x2_comp2) - target_ratio."""
        s1 = streams_dict[self.stream1_name]
        s2 = streams_dict[self.stream2_name]

        flow1 = s1.flow_rate * s1.composition[self.comp1_name]
        flow2 = s2.flow_rate * s2.composition[self.comp2_name]

        if flow2 == 0:
            return float("inf")

        actual_ratio = flow1 / flow2
        return actual_ratio - self.target_ratio

    def validate_references(self, streams_dict):
        """Verify streams and components exist."""
        if self.stream1_name not in streams_dict:
            raise KeyError(f"Stream '{self.stream1_name}' not found")
        if self.stream2_name not in streams_dict:
            raise KeyError(f"Stream '{self.stream2_name}' not found")

        s1 = streams_dict[self.stream1_name]
        s2 = streams_dict[self.stream2_name]

        if self.comp1_name not in s1.composition:
            raise KeyError(
                f"Component '{self.comp1_name}' not in stream '{self.stream1_name}'"
            )
        if self.comp2_name not in s2.composition:
            raise KeyError(
                f"Component '{self.comp2_name}' not in stream '{self.stream2_name}'"
            )

    @property
    def description(self):
        """Return constraint description."""
        return (
            f"ComponentFlowRatio: "
            f"({self.stream1_name}.F × {self.stream1_name}.x_{self.comp1_name}) / "
            f"({self.stream2_name}.F × {self.stream2_name}.x_{self.comp2_name}) = "
            f"{self.target_ratio}"
        )


class CompositionRatio(Ratio):
    """
    Represents a ratio between component compositions in a stream.

    Constraint: stream.composition[comp1] / stream.composition[comp2] = target_ratio

    Example:
        >>> ratio = CompositionRatio("Outlet", "Acetone", "Water", target_ratio=4.0)
        >>> # Constraint: x_acetone / x_water = 4.0
    """

    def __init__(self, stream_name, comp1_name, comp2_name, target_ratio):
        """
        Initialize a composition ratio constraint.

        Parameters:
            stream_name (str): Name of the stream.
            comp1_name (str): Name of numerator component.
            comp2_name (str): Name of denominator component.
            target_ratio (float): Target ratio value.
        """
        self.stream_name = stream_name
        self.comp1_name = comp1_name
        self.comp2_name = comp2_name
        self.target_ratio = target_ratio

    def compute_residual(self, streams_dict):
        """Compute (x_comp1 / x_comp2) - target_ratio."""
        stream = streams_dict[self.stream_name]

        x1 = stream.composition[self.comp1_name]
        x2 = stream.composition[self.comp2_name]

        if x2 == 0:
            return float("inf")

        actual_ratio = x1 / x2
        return actual_ratio - self.target_ratio

    def validate_references(self, streams_dict):
        """Verify stream and components exist."""
        if self.stream_name not in streams_dict:
            raise KeyError(f"Stream '{self.stream_name}' not found")

        stream = streams_dict[self.stream_name]

        if self.comp1_name not in stream.composition:
            raise KeyError(
                f"Component '{self.comp1_name}' not in stream '{self.stream_name}'"
            )
        if self.comp2_name not in stream.composition:
            raise KeyError(
                f"Component '{self.comp2_name}' not in stream '{self.stream_name}'"
            )

    @property
    def description(self):
        """Return constraint description."""
        return (
            f"CompositionRatio: "
            f"{self.stream_name}.x_{self.comp1_name} / "
            f"{self.stream_name}.x_{self.comp2_name} = {self.target_ratio}"
        )
