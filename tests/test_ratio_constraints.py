"""
Test suite for ratio constraint classes.

Tests cover FlowRatio, ComponentFlowRatio, CompositionRatio, and integration
with ProcessUnit for degree-of-freedom calculations and material balance solving.
"""

import numpy as np
import pytest

from src.material_balances import (
    Component,
    ComponentFlowRatio,
    CompositionRatio,
    FlowRatio,
    ProcessUnit,
    Stream,
)


class TestFlowRatio:
    """Test suite for FlowRatio constraint."""

    def test_flow_ratio_initialization(self):
        """Test basic FlowRatio creation."""
        ratio = FlowRatio("stream1", "stream2", target_ratio=0.5)
        assert ratio.stream1_name == "stream1"
        assert ratio.stream2_name == "stream2"
        assert ratio.target_ratio == 0.5

    def test_flow_ratio_residual_zero(self):
        """Test residual is zero when actual ratio equals target."""
        water = Component("Water")

        s1 = Stream("stream1", 10.0, "mole", [water], {"Water": 1.0})
        s2 = Stream("stream2", 20.0, "mole", [water], {"Water": 1.0})

        ratio = FlowRatio("stream1", "stream2", target_ratio=0.5)
        streams_dict = {"stream1": s1, "stream2": s2}

        residual = ratio.compute_residual(streams_dict)
        assert np.isclose(residual, 0.0)

    def test_flow_ratio_residual_nonzero(self):
        """Test residual is nonzero when actual ratio differs from target."""
        water = Component("Water")

        s1 = Stream("stream1", 10.0, "mole", [water], {"Water": 1.0})
        s2 = Stream("stream2", 20.0, "mole", [water], {"Water": 1.0})

        ratio = FlowRatio("stream1", "stream2", target_ratio=0.6)
        streams_dict = {"stream1": s1, "stream2": s2}

        residual = ratio.compute_residual(streams_dict)
        assert np.isclose(residual, 0.5 - 0.6)

    def test_flow_ratio_description(self):
        """Test description property."""
        ratio = FlowRatio("inlet", "outlet", 2.0)
        assert "FlowRatio" in ratio.description
        assert "inlet" in ratio.description
        assert "outlet" in ratio.description

    def test_flow_ratio_validate_references_success(self):
        """Test validation passes with valid references."""
        water = Component("Water")
        s1 = Stream("inlet", 10.0, "mole", [water], {"Water": 1.0})
        s2 = Stream("outlet", 5.0, "mole", [water], {"Water": 1.0})

        ratio = FlowRatio("inlet", "outlet", 2.0)
        streams_dict = {"inlet": s1, "outlet": s2}

        # Should not raise
        ratio.validate_references(streams_dict)

    def test_flow_ratio_validate_references_missing_stream1(self):
        """Test validation fails when stream1 is missing."""
        water = Component("Water")
        s2 = Stream("outlet", 5.0, "mole", [water], {"Water": 1.0})

        ratio = FlowRatio("inlet", "outlet", 2.0)
        streams_dict = {"outlet": s2}

        with pytest.raises(KeyError):
            ratio.validate_references(streams_dict)

    def test_flow_ratio_validate_references_missing_stream2(self):
        """Test validation fails when stream2 is missing."""
        water = Component("Water")
        s1 = Stream("inlet", 10.0, "mole", [water], {"Water": 1.0})

        ratio = FlowRatio("inlet", "outlet", 2.0)
        streams_dict = {"inlet": s1}

        with pytest.raises(KeyError):
            ratio.validate_references(streams_dict)


class TestComponentFlowRatio:
    """Test suite for ComponentFlowRatio constraint."""

    def test_component_flow_ratio_initialization(self):
        """Test basic ComponentFlowRatio creation."""
        ratio = ComponentFlowRatio(
            "stream1", "Water", "stream2", "Salt", target_ratio=2.0
        )
        assert ratio.stream1_name == "stream1"
        assert ratio.comp1_name == "Water"
        assert ratio.stream2_name == "stream2"
        assert ratio.comp2_name == "Salt"
        assert ratio.target_ratio == 2.0

    def test_component_flow_ratio_residual_zero(self):
        """Test residual is zero when actual component flow ratio equals target."""
        water = Component("Water")
        salt = Component("Salt")

        # stream1: 10 kg total, 80% water = 8 kg water
        s1 = Stream(
            "stream1", 10.0, "mass", [water, salt], {"Water": 0.8, "Salt": None}
        )
        # stream2: 4 kg total, 100% salt = 4 kg salt
        s2 = Stream("stream2", 4.0, "mass", [water, salt], {"Water": 0, "Salt": 1.0})

        ratio = ComponentFlowRatio(
            "stream1", "Water", "stream2", "Salt", target_ratio=2.0
        )
        streams_dict = {"stream1": s1, "stream2": s2}

        residual = ratio.compute_residual(streams_dict)
        # (10 * 0.8) / (4 * 1.0) = 8 / 4 = 2.0 - 2.0 = 0
        assert np.isclose(residual, 0.0, atol=1e-6)

    def test_component_flow_ratio_description(self):
        """Test description property."""
        ratio = ComponentFlowRatio("inlet", "Acetone", "outlet", "Water", 3.0)
        assert "ComponentFlowRatio" in ratio.description
        assert "inlet" in ratio.description
        assert "Acetone" in ratio.description

    def test_component_flow_ratio_validate_references_success(self):
        """Test validation passes with valid references."""
        water = Component("Water")
        salt = Component("Salt")

        s1 = Stream(
            "stream1", 10.0, "mass", [water, salt], {"Water": 0.8, "Salt": None}
        )
        s2 = Stream("stream2", 4.0, "mass", [water, salt], {"Water": 0, "Salt": 1.0})

        ratio = ComponentFlowRatio("stream1", "Water", "stream2", "Salt", 2.0)
        streams_dict = {"stream1": s1, "stream2": s2}

        # Should not raise
        ratio.validate_references(streams_dict)

    def test_component_flow_ratio_validate_references_missing_component1(self):
        """Test validation fails when component1 is missing."""
        water = Component("Water")
        salt = Component("Salt")

        s1 = Stream(
            "stream1", 10.0, "mass", [water, salt], {"Water": 0.8, "Salt": None}
        )
        s2 = Stream("stream2", 4.0, "mass", [water, salt], {"Water": 0, "Salt": 1.0})

        ratio = ComponentFlowRatio("stream1", "Acetone", "stream2", "Salt", 2.0)
        streams_dict = {"stream1": s1, "stream2": s2}

        with pytest.raises(KeyError):
            ratio.validate_references(streams_dict)


class TestCompositionRatio:
    """Test suite for CompositionRatio constraint."""

    def test_composition_ratio_initialization(self):
        """Test basic CompositionRatio creation."""
        ratio = CompositionRatio("outlet", "Acetone", "Water", target_ratio=0.75)
        assert ratio.stream_name == "outlet"
        assert ratio.comp1_name == "Acetone"
        assert ratio.comp2_name == "Water"
        assert ratio.target_ratio == 0.75

    def test_composition_ratio_residual_zero(self):
        """Test residual is zero when actual composition ratio equals target."""
        water = Component("Water")
        acetone = Component("Acetone")

        # Stream: 75% Acetone, 25% Water -> ratio = 0.75 / 0.25 = 3.0
        stream = Stream(
            "outlet",
            10.0,
            "mole",
            [water, acetone],
            {"Water": 0.25, "Acetone": 0.75},
        )

        ratio = CompositionRatio("outlet", "Acetone", "Water", target_ratio=3.0)
        streams_dict = {"outlet": stream}

        residual = ratio.compute_residual(streams_dict)
        assert np.isclose(residual, 0.0, atol=1e-6)

    def test_composition_ratio_description(self):
        """Test description property."""
        ratio = CompositionRatio("stream", "Acetone", "Water", 2.0)
        assert "CompositionRatio" in ratio.description
        assert "stream" in ratio.description
        assert "Acetone" in ratio.description

    def test_composition_ratio_validate_references_success(self):
        """Test validation passes with valid references."""
        water = Component("Water")
        acetone = Component("Acetone")

        stream = Stream(
            "outlet", 10.0, "mole", [water, acetone], {"Water": 0.25, "Acetone": None}
        )

        ratio = CompositionRatio("outlet", "Acetone", "Water", 3.0)
        streams_dict = {"outlet": stream}

        # Should not raise
        ratio.validate_references(streams_dict)

    def test_composition_ratio_validate_references_missing_stream(self):
        """Test validation fails when stream is missing."""
        ratio = CompositionRatio("outlet", "Acetone", "Water", 3.0)
        streams_dict = {}

        with pytest.raises(KeyError):
            ratio.validate_references(streams_dict)

    def test_composition_ratio_validate_references_missing_component(self):
        """Test validation fails when component is missing."""
        water = Component("Water")

        stream = Stream("outlet", 10.0, "mole", [water], {"Water": 1.0})

        ratio = CompositionRatio("outlet", "Acetone", "Water", 3.0)
        streams_dict = {"outlet": stream}

        with pytest.raises(KeyError):
            ratio.validate_references(streams_dict)


class TestProcessUnitWithRatios:
    """Test suite for ProcessUnit with ratio constraints."""

    def test_process_unit_with_single_flow_ratio(self):
        """Test ProcessUnit initialization with a single flow ratio."""
        water = Component("Water")

        inlet = Stream("inlet", 10.0, "mole", [water], {"Water": 1.0})
        outlet1 = Stream("outlet1", None, "mole", [water], {"Water": 1.0})
        outlet2 = Stream("outlet2", None, "mole", [water], {"Water": 1.0})

        ratio = FlowRatio("outlet1", "outlet2", 2.0)

        unit = ProcessUnit("Splitter", [inlet], [outlet1, outlet2], ratios=[ratio])
        assert len(unit.ratios) == 1
        assert unit.ratios[0] is ratio

    def test_process_unit_dof_with_ratio(self):
        """Test DOF calculation includes ratio constraints."""
        water = Component("Water")

        inlet = Stream("inlet", 10.0, "mole", [water], {"Water": 1.0})
        outlet1 = Stream("outlet1", None, "mole", [water], {"Water": 1.0})
        outlet2 = Stream("outlet2", None, "mole", [water], {"Water": 1.0})

        ratio = FlowRatio("outlet1", "outlet2", 2.0)

        unit = ProcessUnit("Splitter", [inlet], [outlet1, outlet2], ratios=[ratio])

        # 2 unknowns - 1 equation - 1 ratio = 0 DOF -> solvable
        assert unit.unknowns == 2
        assert unit.independent_material_balances == 1
        assert unit.is_solvable() is True

    def test_process_unit_dof_without_ratio(self):
        """Test that same system is not solvable without ratio."""
        water = Component("Water")

        inlet = Stream("inlet", 10.0, "mole", [water], {"Water": 1.0})
        outlet1 = Stream("outlet1", None, "mole", [water], {"Water": 1.0})
        outlet2 = Stream("outlet2", None, "mole", [water], {"Water": 1.0})

        unit = ProcessUnit("Splitter", [inlet], [outlet1, outlet2])

        # 2 unknowns - 1 equation = 1 DOF -> not solvable
        assert not unit.is_solvable()

    def test_process_unit_validate_ratio_references_fails(self):
        """Test that ProcessUnit validates ratio references."""
        water = Component("Water")

        inlet = Stream("inlet", 10.0, "mole", [water], {"Water": 1.0})
        outlet = Stream("outlet", None, "mole", [water], {"Water": 1.0})

        ratio = FlowRatio("invalid_stream", "outlet", 2.0)

        with pytest.raises(ValueError, match="invalid references"):
            ProcessUnit("Unit", [inlet], [outlet], ratios=[ratio])

    def test_process_unit_with_multiple_ratios(self):
        """Test ProcessUnit with multiple independent ratios."""
        water = Component("Water")

        inlet1 = Stream("inlet1", None, "mole", [water], {"Water": 1.0})
        inlet2 = Stream("inlet2", None, "mole", [water], {"Water": 1.0})
        outlet1 = Stream("outlet1", None, "mole", [water], {"Water": 1.0})
        outlet2 = Stream("outlet2", None, "mole", [water], {"Water": 1.0})

        ratio1 = FlowRatio("inlet1", "inlet2", 1.0)
        ratio2 = FlowRatio("outlet1", "outlet2", 1.5)

        unit = ProcessUnit(
            "Complex",
            [inlet1, inlet2],
            [outlet1, outlet2],
            ratios=[ratio1, ratio2],
        )

        # 4 unknowns - 1 equation - 2 ratios = 1 DOF
        assert unit.unknowns == 4
        assert unit.independent_material_balances == 1
        assert not unit.is_solvable()


class TestJamProductionCase:
    """Test suite for jam production case with component flow ratio."""

    def test_jam_production_setup(self):
        """Test jam production process unit setup."""
        solids = Component("Solids")
        sugar = Component("Sugar")
        water = Component("Water")

        # Strawberry inlet: 15% solids, 85% water
        strawberry = Stream(
            "Strawberry",
            None,
            "mass",
            [solids, sugar, water],
            {"Solids": 0.15, "Sugar": 0, "Water": 0.85},
        )

        # Sugar inlet: pure sugar
        sugar_inlet = Stream(
            "SugarInlet",
            None,
            "mass",
            [solids, sugar, water],
            {"Solids": 0, "Sugar": 1.0, "Water": 0},
        )

        # Jam outlet: known composition (solids 10%, sugar 56.7%, water 33.3%)
        jam = Stream(
            "Jam",
            1.0,
            "mass",
            [solids, sugar, water],
            {"Solids": 0.1, "Sugar": 0.567, "Water": 0.333},
        )

        # Water outlet
        evaporated = Stream(
            "Evaporated",
            None,
            "mass",
            [solids, sugar, water],
            {"Solids": 0, "Sugar": 0, "Water": 1.0},
        )

        # Without ratio: DOF = 3 unknowns - 3 material balances = 0 (solvable)
        heater_no_ratio = ProcessUnit(
            "Heater",
            [strawberry, sugar_inlet],
            [jam, evaporated],
        )
        assert heater_no_ratio.is_solvable()

        # With ratio constraint: DOF = 3 - 3 - 1 = -1 (over-determined)
        ratio = FlowRatio("Strawberry", "SugarInlet", target_ratio=45 / 55)
        heater = ProcessUnit(
            "Heater",
            [strawberry, sugar_inlet],
            [jam, evaporated],
            ratios=[ratio],
        )
        assert not heater.is_solvable()  # Over-determined with ratio

    def test_jam_production_solve(self):
        """Test solving jam production material balance with ratio constraint."""
        water = Component("Water")

        inlet1 = Stream("inlet1", None, "mass", [water], {"Water": 1.0})
        inlet2 = Stream("inlet2", None, "mass", [water], {"Water": 1.0})
        outlet = Stream("outlet", 10.0, "mass", [water], {"Water": 1.0})

        ratio = FlowRatio("inlet1", "inlet2", target_ratio=45 / 55)

        unit = ProcessUnit("Unit", [inlet1, inlet2], [outlet], ratios=[ratio])

        unit.solve_material_balances()

        # Check that all flow rates are now determined
        assert inlet1.flow_rate is not None
        assert inlet2.flow_rate is not None

        # Check flow rates are positive
        assert inlet1.flow_rate > 0
        assert inlet2.flow_rate > 0

        # Check ratio is satisfied
        actual_ratio = inlet1.flow_rate / inlet2.flow_rate
        assert np.isclose(actual_ratio, 45 / 55, rtol=1e-3)

        # Check total mass balance
        total_in = inlet1.flow_rate + inlet2.flow_rate
        assert np.isclose(total_in, outlet.flow_rate, rtol=1e-3)
