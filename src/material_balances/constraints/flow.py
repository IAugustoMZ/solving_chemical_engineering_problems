"""Total-flow constraints."""

from .base import Ratio


class FlowRatio(Ratio):
    """Constrain the ratio of two stream flow rates."""

    def __init__(self, stream1_name, stream2_name, target_ratio, basis=None):
        self.stream1_name = stream1_name
        self.stream2_name = stream2_name
        self.target_ratio = target_ratio
        self.basis = basis

    def compute_residual(self, streams_dict):
        s1 = streams_dict[self.stream1_name]
        s2 = streams_dict[self.stream2_name]
        basis = self.basis or getattr(self, "resolved_basis", None)
        if basis == "mass":
            flow1, flow2 = s1.mass_flow_rate, s2.mass_flow_rate
        elif basis == "molar":
            flow1, flow2 = s1.molar_flow_rate, s2.molar_flow_rate
        else:
            flow1, flow2 = s1.flow_rate, s2.flow_rate
        if flow2 in (None, 0):
            return float("inf")
        return flow1 / flow2 - self.target_ratio

    def validate_references(self, streams_dict):
        if self.stream1_name not in streams_dict:
            raise KeyError(f"Stream '{self.stream1_name}' not found in process unit")
        if self.stream2_name not in streams_dict:
            raise KeyError(f"Stream '{self.stream2_name}' not found in process unit")

    @property
    def description(self):
        return (
            f"FlowRatio: {self.stream1_name}.F / {self.stream2_name}.F = "
            f"{self.target_ratio}"
        )
