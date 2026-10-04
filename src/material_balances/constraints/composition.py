"""Composition constraints."""

from .base import Ratio


class CompositionRatio(Ratio):
    """Constrain the ratio of two component fractions in one stream."""

    def __init__(self, stream_name, comp1_name, comp2_name, target_ratio):
        self.stream_name = stream_name
        self.comp1_name = comp1_name
        self.comp2_name = comp2_name
        self.target_ratio = target_ratio

    def compute_residual(self, streams_dict):
        stream = streams_dict[self.stream_name]
        x1 = stream.composition[self.comp1_name]
        x2 = stream.composition[self.comp2_name]
        if x2 == 0:
            return float("inf")
        return x1 / x2 - self.target_ratio

    def validate_references(self, streams_dict):
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
        return (
            f"CompositionRatio: {self.stream_name}.x_{self.comp1_name} / "
            f"{self.stream_name}.x_{self.comp2_name} = {self.target_ratio}"
        )
