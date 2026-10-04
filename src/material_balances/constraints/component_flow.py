"""Component-flow constraints."""

from .base import Ratio


class ComponentFlowRatio(Ratio):
    """Constrain the ratio of component flows in two streams."""

    def __init__(
        self,
        stream1_name,
        comp1_name,
        stream2_name,
        comp2_name,
        target_ratio,
        basis=None,
    ):
        self.stream1_name = stream1_name
        self.comp1_name = comp1_name
        self.stream2_name = stream2_name
        self.comp2_name = comp2_name
        self.target_ratio = target_ratio
        self.basis = basis

    def compute_residual(self, streams_dict):
        s1 = streams_dict[self.stream1_name]
        s2 = streams_dict[self.stream2_name]
        basis = self.basis or getattr(self, "resolved_basis", None)
        if basis == "mass":
            flow1 = s1.mass_flow_rate * s1.mass_fractions[self.comp1_name]
            flow2 = s2.mass_flow_rate * s2.mass_fractions[self.comp2_name]
        elif basis == "molar":
            flow1 = s1.molar_flow_rate * s1.mole_fractions[self.comp1_name]
            flow2 = s2.molar_flow_rate * s2.mole_fractions[self.comp2_name]
        else:
            flow1 = s1.flow_rate * s1.composition[self.comp1_name]
            flow2 = s2.flow_rate * s2.composition[self.comp2_name]
        if flow2 in (None, 0):
            return float("inf")
        return flow1 / flow2 - self.target_ratio

    def validate_references(self, streams_dict):
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
        return (
            f"ComponentFlowRatio: ({self.stream1_name}.F × "
            f"{self.stream1_name}.x_{self.comp1_name}) / ({self.stream2_name}.F × "
            f"{self.stream2_name}.x_{self.comp2_name}) = {self.target_ratio}"
        )


class ComponentFlowValue(Ratio):
    """Constrain the absolute flow of one component in a stream."""

    def __init__(self, stream_name, comp_name, target_value, basis=None):
        self.stream_name = stream_name
        self.comp_name = comp_name
        self.target_value = target_value
        self.basis = basis

    def compute_residual(self, streams_dict):
        stream = streams_dict[self.stream_name]
        basis = self.basis or getattr(self, "resolved_basis", None)
        if basis == "mass":
            actual_value = stream.mass_flow_rate * stream.mass_fractions[self.comp_name]
        elif basis == "molar":
            actual_value = (
                stream.molar_flow_rate * stream.mole_fractions[self.comp_name]
            )
        else:
            actual_value = stream.flow_rate * stream.composition[self.comp_name]
        return actual_value - self.target_value

    def validate_references(self, streams_dict):
        if self.stream_name not in streams_dict:
            raise KeyError(f"Stream '{self.stream_name}' not found")
        stream = streams_dict[self.stream_name]
        if self.comp_name not in stream.composition:
            raise KeyError(
                f"Component '{self.comp_name}' not in stream '{self.stream_name}'"
            )

    @property
    def description(self):
        return (
            f"ComponentFlowValue: {self.stream_name}.F × "
            f"{self.stream_name}.x_{self.comp_name} = {self.target_value}"
        )
