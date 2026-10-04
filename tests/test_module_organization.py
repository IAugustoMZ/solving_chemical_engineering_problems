"""Regression tests for the modular public and compatibility import paths."""

import numpy as np

from src.material_balances import Component, FlowRatio, Stream, StreamFactory
from src.material_balances.component import Component as ComponentModel
from src.material_balances.constraints.flow import FlowRatio as FlowRatioModel
from src.material_balances.factory import StreamFactory as FactoryModel
from src.material_balances.solver import ComponentFlowSystem
from src.material_balances.stream import StreamFactory as LegacyStreamFactory


def test_public_and_compatibility_imports_reference_the_same_models():
    """Existing notebook imports and the modular paths must remain equivalent."""
    assert Component is ComponentModel
    assert FlowRatio is FlowRatioModel
    assert StreamFactory is FactoryModel
    assert LegacyStreamFactory is FactoryModel
    assert Stream.__module__ == "src.material_balances.stream"


def test_component_flow_system_exposes_structural_analysis():
    """Structural rank belongs to the solver-system value object."""
    system = ComponentFlowSystem(
        variable_refs=[("feed", "water"), ("product", "water")],
        matrix=np.array([[1.0, -1.0], [1.0, 0.0]]),
        targets=np.array([0.0, 2.0]),
        labels=["material balance: water", "specified component flow"],
    )

    assert system.rank == 2
    assert system.degrees_of_freedom == 0
    assert system.is_uniquely_determined
