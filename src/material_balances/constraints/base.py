"""Contracts shared by algebraic material-balance constraints."""

from abc import ABC, abstractmethod


class Ratio(ABC):
    """An algebraic relation between streams or component flows."""

    @abstractmethod
    def compute_residual(self, streams_dict):
        """Return the constraint residual for fully specified streams."""

    @abstractmethod
    def validate_references(self, streams_dict):
        """Raise an exception when a referenced stream or component is absent."""

    @property
    @abstractmethod
    def description(self):
        """A human-readable representation of the constraint."""
