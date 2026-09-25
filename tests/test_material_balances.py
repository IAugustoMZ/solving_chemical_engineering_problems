"""
Comprehensive test suite for material_balances module.

Tests cover Components, Streams, ProcessUnits, and utility solver functions
with focus on edge cases, error handling, and core functionality.
"""

import pytest
import numpy as np
from src.material_balances import (
    Component,
    Stream,
    ProcessUnit
)


class TestComponent:
    """Test suite for Component class."""

    def test_component_initialization(self):
        """Test basic component creation with valid name."""
        comp = Component("Water")
        assert comp.name == "Water"

    def test_component_various_names(self):
        """Test component creation with different types of valid names."""
        names = ["Acetone", "CO2", "H2O", "Sugar", "NaCl"]
        for name in names:
            comp = Component(name)
            assert comp.name == name

    def test_component_name_with_spaces(self):
        """Test component name can contain spaces."""
        comp = Component("Sodium Chloride")
        assert comp.name == "Sodium Chloride"

    def test_component_name_attribute(self):
        """Test accessing component name attribute."""
        comp = Component("Water")
        assert hasattr(comp, "name")
        assert comp.name == "Water"


class TestStream:
    """Test suite for Stream class."""

    def test_stream_basic_initialization(self):
        """Test basic stream creation with all parameters specified."""
        water = Component("Water")
        acetone = Component("Acetone")
        stream = Stream(
            name="feed",
            flow_rate=10.0,
            flow_type="mole",
            components=[water, acetone],
            composition={"Water": 0.65, "Acetone": 0.35},
        )
        assert stream.name == "feed"
        assert stream.flow_rate == 10.0
        assert stream.flow_type == "mole"
        assert len(stream.components) == 2

    def test_stream_with_none_flow_rate(self):
        """Test stream with unknown flow rate (None)."""
        water = Component("Water")
        stream = Stream(
            name="product",
            flow_rate=None,
            flow_type="mole",
            components=[water],
            composition={"Water": 1.0},
        )
        assert stream.flow_rate is None

    def test_stream_complementary_composition_single_none(self):
        """Test automatic calculation of missing composition value."""
        water = Component("Water")
        acetone = Component("Acetone")
        stream = Stream(
            name="feed",
            flow_rate=10.0,
            flow_type="mole",
            components=[water, acetone],
            composition={"Water": 0.65, "Acetone": None},
        )
        assert np.isclose(stream.composition["Acetone"], 0.35)

    def test_stream_complementary_composition_zero(self):
        """Test calculation of missing composition when sum is 1.0."""
        water = Component("Water")
        acetone = Component("Acetone")
        stream = Stream(
            name="liquid",
            flow_rate=10.0,
            flow_type="mole",
            components=[water, acetone],
            composition={"Water": 1.0, "Acetone": None},
        )
        assert np.isclose(stream.composition["Acetone"], 0.0)

    def test_stream_validation_composition_sum(self):
        """Test that composition values must sum to 1.0."""
        water = Component("Water")
        acetone = Component("Acetone")
        with pytest.raises(ValueError, match="sum to 1"):
            Stream(
                name="bad_stream",
                flow_rate=10.0,
                flow_type="mole",
                components=[water, acetone],
                composition={"Water": 0.6, "Acetone": 0.3},  # sums to 0.9
            )

    def test_stream_validation_components_mismatch(self):
        """Test that number of components must match composition dict."""
        water = Component("Water")
        acetone = Component("Acetone")
        with pytest.raises(ValueError, match="Number of components"):
            Stream(
                name="bad_stream",
                flow_rate=10.0,
                flow_type="mole",
                components=[water, acetone],
                composition={"Water": 1.0},  # only 1 component in composition
            )

    def test_stream_validation_multiple_none_compositions(self):
        """Test error when more than one composition value is None."""
        water = Component("Water")
        acetone = Component("Acetone")
        with pytest.raises(ValueError, match="more than one"):
            Stream(
                name="bad_stream",
                flow_rate=10.0,
                flow_type="mole",
                components=[water, acetone],
                composition={"Water": None, "Acetone": None},
            )

    def test_stream_direction_input(self):
        """Test stream with input direction."""
        water = Component("Water")
        stream = Stream(
            name="feed",
            flow_rate=10.0,
            flow_type="mole",
            components=[water],
            composition={"Water": 1.0},
            direction="input",
        )
        assert stream.direction == "input"

    def test_stream_direction_output(self):
        """Test stream with output direction."""
        water = Component("Water")
        stream = Stream(
            name="product",
            flow_rate=10.0,
            flow_type="mole",
            components=[water],
            composition={"Water": 1.0},
            direction="output",
        )
        assert stream.direction == "output"

    def test_stream_different_flow_types(self):
        """Test stream with different flow types."""
        water = Component("Water")
        for flow_type in ["mole", "mass", "volumetric"]:
            stream = Stream(
                name="test",
                flow_rate=10.0,
                flow_type=flow_type,
                components=[water],
                composition={"Water": 1.0},
            )
            assert stream.flow_type == flow_type

    def test_stream_single_component(self):
        """Test stream with single component."""
        pure_water = Component("Water")
        stream = Stream(
            name="pure_water",
            flow_rate=100.0,
            flow_type="mass",
            components=[pure_water],
            composition={"Water": 1.0},
        )
        assert len(stream.components) == 1
        assert stream.composition["Water"] == 1.0

    def test_stream_three_component_system(self):
        """Test stream with three components."""
        water = Component("Water")
        ethanol = Component("Ethanol")
        acetone = Component("Acetone")
        stream = Stream(
            name="ternary_mixture",
            flow_rate=10.0,
            flow_type="mole",
            components=[water, ethanol, acetone],
            composition={"Water": 0.5, "Ethanol": 0.3, "Acetone": None},
        )
        assert np.isclose(stream.composition["Acetone"], 0.2)


class TestProcessUnit:
    """Test suite for ProcessUnit class."""

    def test_process_unit_basic_initialization(self):
        """Test basic process unit creation."""
        water = Component("Water")
        acetone = Component("Acetone")

        feed = Stream(
            "feed", 10.0, "mole",
            [water, acetone],
            {"Water": 0.65, "Acetone": None}
        )
        vapor = Stream(
            "vapor", None, "mole",
            [water, acetone],
            {"Water": 0.25, "Acetone": None}
        )
        liquid = Stream(
            "liquid", None, "mole",
            [water, acetone],
            {"Water": 0.813, "Acetone": None}
        )

        unit = ProcessUnit("Evaporator", [feed], [vapor, liquid])
        assert unit.name == "Evaporator"
        assert len(unit.input_streams) == 1
        assert len(unit.output_streams) == 2

    def test_process_unit_component_validation(self):
        """Test that inlet and outlet must have same components."""
        water = Component("Water")
        acetone = Component("Acetone")
        ethanol = Component("Ethanol")

        feed = Stream(
            "feed", 10.0, "mole",
            [water, acetone],
            {"Water": 0.65, "Acetone": None}
        )
        product = Stream(
            "product", None, "mole",
            [water, ethanol],  # different components
            {"Water": 0.5, "Ethanol": None}
        )

        with pytest.raises(ValueError, match="same components"):
            ProcessUnit("BadUnit", [feed], [product])

    def test_process_unit_degree_of_freedom_calculation(self):
        """Test that degree of freedom is calculated correctly."""
        water = Component("Water")
        acetone = Component("Acetone")

        feed = Stream(
            "feed", 10.0, "mole",
            [water, acetone],
            {"Water": 0.65, "Acetone": None}
        )
        vapor = Stream(
            "vapor", None, "mole",
            [water, acetone],
            {"Water": 0.25, "Acetone": None}
        )
        liquid = Stream(
            "liquid", None, "mole",
            [water, acetone],
            {"Water": 0.813, "Acetone": None}
        )

        unit = ProcessUnit("Evaporator", [feed], [vapor, liquid])
        # 2 components = 2 independent material balances
        assert unit.independent_material_balances == 2
        # 2 unknown flow rates (vapor, liquid) = 2 unknowns
        assert unit.unknowns == 2
        # 2 unknowns <= 2 equations, so solvable
        assert unit.is_solvable()

    def test_process_unit_is_solvable_true(self):
        """Test solvability check returns True when system is solvable."""
        water = Component("Water")
        acetone = Component("Acetone")

        feed = Stream(
            "feed", 10.0, "mole",
            [water, acetone],
            {"Water": 0.65, "Acetone": None}
        )
        vapor = Stream(
            "vapor", None, "mole",
            [water, acetone],
            {"Water": 0.75, "Acetone": None}
        )
        liquid = Stream(
            "liquid", None, "mole",
            [water, acetone],
            {"Water": 0.187, "Acetone": None}
        )

        unit = ProcessUnit("Evaporator", [feed], [vapor, liquid])
        assert unit.is_solvable() is True

    def test_process_unit_is_solvable_false(self):
        """Test solvability check returns False when under-specified."""
        water = Component("Water")

        inlet = Stream(
            "inlet", 10.0, "mole",
            [water],
            {"Water": 1.0}
        )
        outlet1 = Stream(
            "outlet1", None, "mole",
            [water],
            {"Water": 1.0}
        )
        outlet2 = Stream(
            "outlet2", None, "mole",
            [water],
            {"Water": 1.0}
        )

        unit = ProcessUnit("Splitter", [inlet], [outlet1, outlet2])
        assert unit.is_solvable() is False

    def test_process_unit_solve_material_balances(self):
        """Test solving material balances for evaporator problem."""
        water = Component("Water")
        acetone = Component("Acetone")

        feed = Stream(
            "feed", 10.0, "mole",
            [water, acetone],
            {"Water": 0.65, "Acetone": None}
        )
        vapor = Stream(
            "vapor", None, "mole",
            [water, acetone],
            {"Water": 0.75, "Acetone": None}
        )
        liquid = Stream(
            "liquid", None, "mole",
            [water, acetone],
            {"Water": 0.187, "Acetone": None}
        )

        unit = ProcessUnit("Evaporator", [feed], [vapor, liquid])
        unit.solve_material_balances()

        # Check that flow rates were solved
        assert vapor.flow_rate is not None
        assert liquid.flow_rate is not None
        assert vapor.flow_rate > 0
        assert liquid.flow_rate > 0

        # Check that total flow is conserved
        assert np.isclose(
            vapor.flow_rate + liquid.flow_rate,
            feed.flow_rate,
            atol=1e-4
        )

    def test_process_unit_solve_unsolvable_raises_error(self):
        """Test that solving unsolvable system raises ValueError."""
        water = Component("Water")

        inlet = Stream(
            "inlet", 10.0, "mole",
            [water],
            {"Water": 1.0}
        )
        outlet1 = Stream(
            "outlet1", None, "mole",
            [water],
            {"Water": 1.0}
        )
        outlet2 = Stream(
            "outlet2", None, "mole",
            [water],
            {"Water": 1.0}
        )

        unit = ProcessUnit("Splitter", [inlet], [outlet1, outlet2])

        with pytest.raises(ValueError, match="cannot be solved"):
            unit.solve_material_balances()

    def test_process_unit_material_balance_residuals(self):
        """Test that material balances are satisfied after solving."""
        water = Component("Water")
        acetone = Component("Acetone")

        feed = Stream(
            "feed", 10.0, "mole",
            [water, acetone],
            {"Water": 0.65, "Acetone": None}
        )
        vapor = Stream(
            "vapor", None, "mole",
            [water, acetone],
            {"Water": 0.75, "Acetone": None}
        )
        liquid = Stream(
            "liquid", None, "mole",
            [water, acetone],
            {"Water": 0.187, "Acetone": None}
        )

        unit = ProcessUnit("Evaporator", [feed], [vapor, liquid])
        unit.solve_material_balances()

        # Check water balance: input = output
        water_in = feed.flow_rate * feed.composition["Water"]
        water_out = (
            vapor.flow_rate * vapor.composition["Water"] +
            liquid.flow_rate * liquid.composition["Water"]
        )
        assert np.isclose(water_in, water_out, atol=1e-6)

        # Check acetone balance: input = output
        acetone_in = feed.flow_rate * feed.composition["Acetone"]
        acetone_out = (
            vapor.flow_rate * vapor.composition["Acetone"] +
            liquid.flow_rate * liquid.composition["Acetone"]
        )
        assert np.isclose(acetone_in, acetone_out, atol=1e-6)

    def test_process_unit_multiple_input_streams(self):
        """Test process unit with multiple input streams (mixer)."""
        water = Component("Water")

        stream1 = Stream(
            "inlet1", 5.0, "mole",
            [water],
            {"Water": 1.0}
        )
        stream2 = Stream(
            "inlet2", 3.0, "mole",
            [water],
            {"Water": 1.0}
        )
        outlet = Stream(
            "outlet", None, "mole",
            [water],
            {"Water": 1.0}
        )

        mixer = ProcessUnit("Mixer", [stream1, stream2], [outlet])
        mixer.solve_material_balances()

        # Output should be sum of inputs
        assert np.isclose(outlet.flow_rate, 8.0, atol=1e-6)

    def test_process_unit_multiple_output_streams(self):
        """Test process unit with multiple output streams (splitter)."""
        water = Component("Water")

        inlet = Stream(
            "inlet", 10.0, "mole",
            [water],
            {"Water": 1.0}
        )
        outlet1 = Stream(
            "outlet1", None, "mole",
            [water],
            {"Water": 1.0}
        )
        outlet2 = Stream(
            "outlet2", None, "mole",
            [water],
            {"Water": 1.0}
        )

        splitter = ProcessUnit("Splitter", [inlet], [outlet1, outlet2])
        # This system is underdetermined (2 unknowns, 1 equation)
        # So it should not be solvable without additional constraints
        assert not splitter.is_solvable()