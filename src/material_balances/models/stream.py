from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Stream:
    """
    Represents a single process stream in a material balance system.

    A stream carries one or more components with specified (or unknown) fractions.
    Fractions may represent molar or mass fractions depending on the problem context.

    Attributes:
        name (str): Unique identifier for the stream (e.g., 'Feed', 'Stream 1').
        direction (str): Flow direction relative to the system boundary.
            Must be 'input' or 'output'.
        total_flow (Optional[float]): Total flow rate of the stream in the
            user-defined units (e.g., kg/h or mol/min). Use None if unknown.
        fractions (Dict[str, Optional[float]]): Mapping of component name to its
            fraction (molar or mass) in the stream. Values must be in [0, 1].
            Use None for components whose fraction is unknown.

    Example:
        >>> feed = Stream(
        ...     name='Feed',
        ...     direction='input',
        ...     total_flow=5300.0,
        ...     fractions={'A': None, 'B': None, 'C': None},
        ... )
    """

    name: str
    direction: str
    total_flow: Optional[float]
    fractions: Dict[str, Optional[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validates that direction is a legal value immediately on construction."""
        if self.direction not in ('input', 'output'):
            raise ValueError(
                f"Stream '{self.name}': direction must be 'input' or 'output', "
                f"got '{self.direction}'."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_input(self) -> bool:
        """True if the stream flows into the system boundary."""
        return self.direction == 'input'

    @property
    def is_output(self) -> bool:
        """True if the stream flows out of the system boundary."""
        return self.direction == 'output'

    @property
    def sign(self) -> float:
        """
        Sign convention for component balance equations.

        Returns +1.0 for input streams (positive contribution to accumulation)
        and -1.0 for output streams (negative contribution).
        """
        return 1.0 if self.is_input else -1.0

    # ------------------------------------------------------------------
    # Fraction helpers
    # ------------------------------------------------------------------

    def n_known_fractions(self, components: List[str]) -> int:
        """
        Counts non-None fractions for the given ordered component list.

        Args:
            components: Full ordered list of component names in the system.

        Returns:
            int: Number of components with a known (non-None) fraction in this stream.
        """
        return sum(1 for c in components if self.fractions.get(c) is not None)

    def n_unknown_fractions(self, components: List[str]) -> int:
        """
        Counts None fractions for the given ordered component list.

        Args:
            components: Full ordered list of component names in the system.

        Returns:
            int: Number of components with an unknown (None) fraction in this stream.
        """
        return sum(1 for c in components if self.fractions.get(c) is None)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_report_dict(self, components: List[str]) -> dict:
        """
        Builds a clean, serialisable dictionary representation of this stream.

        All numpy scalars are cast to plain Python floats to avoid serialisation
        issues. Component flows are computed as total_flow × fraction when both
        values are available.

        Args:
            components: Ordered list of component names for the entire system.

        Returns:
            dict: A dictionary with the following keys:
                - ``'direction'``: 'input' or 'output'.
                - ``'total_flow'``: float or None.
                - ``'fractions'``: dict mapping each component to float or None.
                - ``'component_flows'``: dict mapping each component to its flow
                  (float) or None if total_flow or the fraction is unknown.
        """
        flow = float(self.total_flow) if self.total_flow is not None else None
        fractions_out: Dict[str, Optional[float]] = {
            c: (float(self.fractions[c]) if self.fractions.get(c) is not None else None)
            for c in components
        }
        component_flows: Dict[str, Optional[float]] = {
            c: float(flow * fractions_out[c])
            if flow is not None and fractions_out[c] is not None
            else None
            for c in components
        }
        return {
            'direction': self.direction,
            'total_flow': flow,
            'fractions': fractions_out,
            'component_flows': component_flows,
        }

