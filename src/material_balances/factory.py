"""Factories for constructing coherent sets of material streams."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .component import Component
from .constraints import (
    ComponentFlowRatio,
    ComponentFlowValue,
    CompositionRatio,
    FlowRatio,
)

if TYPE_CHECKING:
    from .stream import Stream


class StreamFactory:
    """Create related streams sharing components and registered constraints."""

    def __init__(
        self,
        component_names: list,
        default_flow_type: str = None,
        molar_masses: dict | list | tuple = None,
    ) -> None:
        self.component_names = component_names
        molar_mass_by_name = self._normalise_molar_masses(molar_masses)
        unknown_names = set(molar_mass_by_name) - set(component_names)
        if unknown_names:
            raise ValueError(
                "Molar masses were provided for unknown components: "
                + ", ".join(sorted(map(str, unknown_names)))
            )
        self.components = [
            Component(name, molar_mass_by_name.get(name)) for name in component_names
        ]
        self.default_flow_type = default_flow_type
        self.streams: dict[str, "Stream"] = {}
        self.ratios: list = []

    def _normalise_molar_masses(self, molar_masses):
        if molar_masses is None:
            return {}
        if isinstance(molar_masses, Mapping):
            return dict(molar_masses)
        if isinstance(molar_masses, (list, tuple)):
            if len(molar_masses) != len(self.component_names):
                raise ValueError(
                    "Molar-mass length "
                    f"({len(molar_masses)}) must match component count "
                    f"({len(self.component_names)})."
                )
            return dict(zip(self.component_names, molar_masses))
        raise TypeError(
            "molar_masses must be a mapping or a list aligned with component_names."
        )

    def add_stream(
        self,
        name: str,
        compositions: list,
        flow_rate: float = None,
        flow_type: str = None,
        direction: str = "input",
        composition_basis: str = None,
    ) -> "Stream":
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
        from .stream import Stream as StreamModel

        stream = StreamModel(
            name=name,
            flow_rate=flow_rate,
            flow_type=resolved_flow_type,
            components=self.components,
            composition=dict(zip(self.component_names, compositions)),
            direction=direction,
            composition_basis=composition_basis,
        )
        self.streams[name] = stream
        return stream

    def get_stream(self, name: str) -> "Stream":
        return self.streams[name]

    def add_ratio(self, ratio_type="flow", **kwargs):
        constructors = {
            "flow": lambda: FlowRatio(
                kwargs["stream1"],
                kwargs["stream2"],
                kwargs["target_ratio"],
                basis=kwargs.get("basis"),
            ),
            "composition": lambda: CompositionRatio(
                kwargs["stream"],
                kwargs["comp1"],
                kwargs["comp2"],
                kwargs["target_ratio"],
            ),
            "component_flow": lambda: ComponentFlowRatio(
                kwargs["stream1"],
                kwargs["comp1"],
                kwargs["stream2"],
                kwargs["comp2"],
                kwargs["target_ratio"],
                basis=kwargs.get("basis"),
            ),
            "component_flow_value": lambda: ComponentFlowValue(
                kwargs["stream"],
                kwargs["comp"],
                kwargs["target_value"],
                basis=kwargs.get("basis"),
            ),
        }
        if ratio_type not in constructors:
            raise ValueError(
                f"Unknown ratio_type '{ratio_type}'. Choose from: 'flow', "
                "'composition', 'component_flow', 'component_flow_value'."
            )
        ratio = constructors[ratio_type]()
        try:
            ratio.validate_references(self.streams)
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"Ratio ({ratio.description}) has invalid references: {error}"
            ) from error
        self.ratios.append(ratio)
        return ratio

    def build_process_unit(self, name: str):
        """Build a process unit from the factory's registered streams."""
        from .process_unit import ProcessUnit

        input_streams = [s for s in self.streams.values() if s.direction == "input"]
        output_streams = [s for s in self.streams.values() if s.direction == "output"]
        return ProcessUnit(name, input_streams, output_streams, ratios=self.ratios)
