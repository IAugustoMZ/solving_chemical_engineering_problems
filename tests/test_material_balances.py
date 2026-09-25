"""
Comprehensive test suite for material_balances module.

Tests cover Components, Streams, ProcessUnits, and utility solver functions
with focus on edge cases, error handling, and core functionality.
"""

import numpy as np
import pytest

from src.material_balances import Component, ProcessUnit, Stream, StreamFactory


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
            "feed", 10.0, "mole", [water, acetone], {"Water": 0.65, "Acetone": None}
        )
        vapor = Stream(
            "vapor", None, "mole", [water, acetone], {"Water": 0.25, "Acetone": None}
        )
        liquid = Stream(
            "liquid", None, "mole", [water, acetone], {"Water": 0.813, "Acetone": None}
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
            "feed", 10.0, "mole", [water, acetone], {"Water": 0.65, "Acetone": None}
        )
        product = Stream(
            "product",
            None,
            "mole",
            [water, ethanol],  # different components
            {"Water": 0.5, "Ethanol": None},
        )

        with pytest.raises(ValueError, match="same components"):
            ProcessUnit("BadUnit", [feed], [product])

    def test_process_unit_degree_of_freedom_calculation(self):
        """Test that degree of freedom is calculated correctly."""
        water = Component("Water")
        acetone = Component("Acetone")

        feed = Stream(
            "feed", 10.0, "mole", [water, acetone], {"Water": 0.65, "Acetone": None}
        )
        vapor = Stream(
            "vapor", None, "mole", [water, acetone], {"Water": 0.25, "Acetone": None}
        )
        liquid = Stream(
            "liquid", None, "mole", [water, acetone], {"Water": 0.813, "Acetone": None}
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
            "feed", 10.0, "mole", [water, acetone], {"Water": 0.65, "Acetone": None}
        )
        vapor = Stream(
            "vapor", None, "mole", [water, acetone], {"Water": 0.75, "Acetone": None}
        )
        liquid = Stream(
            "liquid", None, "mole", [water, acetone], {"Water": 0.187, "Acetone": None}
        )

        unit = ProcessUnit("Evaporator", [feed], [vapor, liquid])
        assert unit.is_solvable() is True

    def test_process_unit_is_solvable_false(self):
        """Test solvability check returns False when under-specified."""
        water = Component("Water")

        inlet = Stream("inlet", 10.0, "mole", [water], {"Water": 1.0})
        outlet1 = Stream("outlet1", None, "mole", [water], {"Water": 1.0})
        outlet2 = Stream("outlet2", None, "mole", [water], {"Water": 1.0})

        unit = ProcessUnit("Splitter", [inlet], [outlet1, outlet2])
        assert unit.is_solvable() is False

    def test_process_unit_solve_material_balances(self):
        """Test solving material balances for evaporator problem."""
        water = Component("Water")
        acetone = Component("Acetone")

        feed = Stream(
            "feed", 10.0, "mole", [water, acetone], {"Water": 0.65, "Acetone": None}
        )
        vapor = Stream(
            "vapor", None, "mole", [water, acetone], {"Water": 0.75, "Acetone": None}
        )
        liquid = Stream(
            "liquid", None, "mole", [water, acetone], {"Water": 0.187, "Acetone": None}
        )

        unit = ProcessUnit("Evaporator", [feed], [vapor, liquid])
        unit.solve_material_balances()

        # Check that flow rates were solved
        assert vapor.flow_rate is not None
        assert liquid.flow_rate is not None
        assert vapor.flow_rate > 0
        assert liquid.flow_rate > 0

        # Check that total flow is conserved
        assert np.isclose(vapor.flow_rate + liquid.flow_rate, feed.flow_rate, atol=1e-4)

    def test_process_unit_solve_unsolvable_raises_error(self):
        """Test that solving unsolvable system raises ValueError."""
        water = Component("Water")

        inlet = Stream("inlet", 10.0, "mole", [water], {"Water": 1.0})
        outlet1 = Stream("outlet1", None, "mole", [water], {"Water": 1.0})
        outlet2 = Stream("outlet2", None, "mole", [water], {"Water": 1.0})

        unit = ProcessUnit("Splitter", [inlet], [outlet1, outlet2])

        with pytest.raises(ValueError, match="cannot be solved"):
            unit.solve_material_balances()

    def test_process_unit_material_balance_residuals(self):
        """Test that material balances are satisfied after solving."""
        water = Component("Water")
        acetone = Component("Acetone")

        feed = Stream(
            "feed", 10.0, "mole", [water, acetone], {"Water": 0.65, "Acetone": None}
        )
        vapor = Stream(
            "vapor", None, "mole", [water, acetone], {"Water": 0.75, "Acetone": None}
        )
        liquid = Stream(
            "liquid", None, "mole", [water, acetone], {"Water": 0.187, "Acetone": None}
        )

        unit = ProcessUnit("Evaporator", [feed], [vapor, liquid])
        unit.solve_material_balances()

        # Check water balance: input = output
        water_in = feed.flow_rate * feed.composition["Water"]
        water_out = (
            vapor.flow_rate * vapor.composition["Water"]
            + liquid.flow_rate * liquid.composition["Water"]
        )
        assert np.isclose(water_in, water_out, atol=1e-6)

        # Check acetone balance: input = output
        acetone_in = feed.flow_rate * feed.composition["Acetone"]
        acetone_out = (
            vapor.flow_rate * vapor.composition["Acetone"]
            + liquid.flow_rate * liquid.composition["Acetone"]
        )
        assert np.isclose(acetone_in, acetone_out, atol=1e-6)

    def test_process_unit_multiple_input_streams(self):
        """Test process unit with multiple input streams (mixer)."""
        water = Component("Water")

        stream1 = Stream("inlet1", 5.0, "mole", [water], {"Water": 1.0})
        stream2 = Stream("inlet2", 3.0, "mole", [water], {"Water": 1.0})
        outlet = Stream("outlet", None, "mole", [water], {"Water": 1.0})

        mixer = ProcessUnit("Mixer", [stream1, stream2], [outlet])
        mixer.solve_material_balances()

        # Output should be sum of inputs
        assert np.isclose(outlet.flow_rate, 8.0, atol=1e-6)

    def test_process_unit_multiple_output_streams(self):
        """Test process unit with multiple output streams (splitter)."""
        water = Component("Water")

        inlet = Stream("inlet", 10.0, "mole", [water], {"Water": 1.0})
        outlet1 = Stream("outlet1", None, "mole", [water], {"Water": 1.0})
        outlet2 = Stream("outlet2", None, "mole", [water], {"Water": 1.0})

        splitter = ProcessUnit("Splitter", [inlet], [outlet1, outlet2])
        # This system is underdetermined (2 unknowns, 1 equation)
        # So it should not be solvable without additional constraints
        assert not splitter.is_solvable()


class TestStreamFactory:
    """Test suite for StreamFactory class."""

    def test_factory_initialization(self):
        """Test basic factory creation with component list."""
        factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
        assert factory.component_names == ["Water", "Acetone"]
        assert len(factory.components) == 2
        assert factory.components[0].name == "Water"
        assert factory.components[1].name == "Acetone"
        assert factory.default_flow_type == "mole"
        assert factory.streams == {}
        assert factory.ratios == []

    def test_add_stream_basic(self):
        """Test creating a stream with positional compositions."""
        factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
        stream = factory.add_stream("Feed", [0.65, 0.35], flow_rate=10.0)

        assert stream.name == "Feed"
        assert stream.flow_rate == 10.0
        assert stream.flow_type == "mole"
        assert stream.composition == {"Water": 0.65, "Acetone": 0.35}
        assert "Feed" in factory.streams
        assert factory.get_stream("Feed") == stream

    def test_add_stream_with_none_composition(self):
        """Test stream creation with one None composition (auto-calculated)."""
        factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
        stream = factory.add_stream("Feed", [0.65, None], flow_rate=10.0)

        assert stream.composition["Water"] == 0.65
        assert np.isclose(stream.composition["Acetone"], 0.35)

    def test_add_stream_duplicate_name_raises(self):
        """Test that duplicate stream name raises ValueError."""
        factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
        factory.add_stream("Feed", [0.65, 0.35], flow_rate=10.0)

        with pytest.raises(ValueError, match="already registered"):
            factory.add_stream("Feed", [0.25, 0.75], flow_rate=5.0)

    def test_add_stream_wrong_composition_length_raises(self):
        """Test that composition length mismatch raises ValueError."""
        factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")

        with pytest.raises(ValueError, match="Composition length"):
            factory.add_stream("Feed", [0.65], flow_rate=10.0)

        with pytest.raises(ValueError, match="Composition length"):
            factory.add_stream("Feed", [0.65, 0.25, 0.1], flow_rate=10.0)

    def test_add_stream_missing_flow_type_raises(self):
        """Test that missing flow_type raises ValueError when no default."""
        factory = StreamFactory(["Water", "Acetone"])
        # No default_flow_type set

        with pytest.raises(ValueError, match="flow_type must be provided"):
            factory.add_stream("Feed", [0.65, 0.35], flow_rate=10.0)

    def test_add_stream_override_flow_type(self):
        """Test overriding default flow_type per stream."""
        factory = StreamFactory(
            ["Water", "Acetone"], default_flow_type="mole"
        )
        stream = factory.add_stream(
            "Feed", [0.65, 0.35], flow_rate=10.0, flow_type="mass"
        )

        assert stream.flow_type == "mass"

    def test_add_stream_with_output_direction(self):
        """Test creating output streams with direction='output'."""
        factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
        vapor = factory.add_stream("Vapor", [0.75, None], direction="output")

        assert vapor.direction == "output"

    def test_get_stream_unknown_name_raises(self):
        """Test that retrieving unknown stream raises KeyError."""
        factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")

        with pytest.raises(KeyError):
            factory.get_stream("NonExistent")

    def test_add_ratio_valid_references(self):
        """Test adding a flow ratio with valid stream references."""
        factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
        factory.add_stream("Feed", [0.65, 0.35], flow_rate=None)
        factory.add_stream("Vapor", [0.75, None], flow_rate=None, direction="output")

        returned_ratio = factory.add_ratio("flow", stream1="Feed", stream2="Vapor", target_ratio=3.45)

        assert returned_ratio is not None
        assert len(factory.ratios) == 1
        assert factory.ratios[0] in factory.ratios

    def test_add_ratio_invalid_stream_raises(self):
        """Test that adding a ratio with unregistered stream raises ValueError."""
        factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
        factory.add_stream("Feed", [0.65, 0.35], flow_rate=10.0)

        with pytest.raises(ValueError, match="invalid references"):
            factory.add_ratio("flow", stream1="Feed", stream2="NonExistent", target_ratio=3.45)

    def test_add_ratio_invalid_component_raises(self):
        """Test that adding a ratio with unregistered component raises ValueError."""
        factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
        factory.add_stream("Feed", [0.65, None], flow_rate=10.0)

        with pytest.raises(ValueError, match="invalid references"):
            factory.add_ratio("composition", stream="Feed", comp1="NonExistent", comp2="Water", target_ratio=2.0)

    def test_add_ratio_composition_ratio(self):
        """Test adding a composition ratio (comp1 / comp2 in a stream)."""
        factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
        factory.add_stream("Outlet", [None, 0.75], flow_rate=10.0)

        ratio = factory.add_ratio("composition", stream="Outlet", comp1="Acetone", comp2="Water", target_ratio=3.0)

        assert ratio is not None
        assert len(factory.ratios) == 1

    def test_add_ratio_component_flow_ratio(self):
        """Test adding a component flow ratio (comp1_flow / comp2_flow)."""
        factory = StreamFactory(["Solids", "Water"], default_flow_type="kg/h")
        factory.add_stream("Stream1", [0.15, None], flow_rate=100.0)
        factory.add_stream("Stream2", [1.0, 0.0], flow_rate=50.0)

        ratio = factory.add_ratio(
            "component_flow",
            stream1="Stream1",
            comp1="Solids",
            stream2="Stream2",
            comp2="Solids",
            target_ratio=0.3,
        )

        assert ratio is not None
        assert len(factory.ratios) == 1

    def test_add_ratio_unknown_type_raises(self):
        """Test that unknown ratio_type raises ValueError."""
        factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
        factory.add_stream("Stream", [0.5, None])

        with pytest.raises(ValueError, match="Unknown ratio_type"):
            factory.add_ratio("unknown_type", stream="Stream", comp1="Water", comp2="Acetone", target_ratio=1.0)

    def test_build_process_unit_simple_evaporator(self):
        """Test building a solvable ProcessUnit for acetone/water evaporator."""
        factory = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
        feed = factory.add_stream("Feed", [0.65, 0.35], flow_rate=10.0)
        vapor = factory.add_stream("Vapor", [0.25, None], direction="output")
        liquid = factory.add_stream("Liquid", [0.813, None], direction="output")

        unit = factory.build_process_unit("H-101")

        assert unit.name == "H-101"
        assert feed in unit.input_streams
        assert vapor in unit.output_streams
        assert liquid in unit.output_streams
        assert unit.is_solvable()

        unit.solve_material_balances()
        unit.print_report()

    def test_build_process_unit_with_ratios(self):
        """Test building a ProcessUnit that includes registered ratios."""
        factory = StreamFactory(
            ["Solids", "Water"], default_flow_type="kg/h"
        )
        strawberry = factory.add_stream("Strawberry", [0.15, None], flow_rate=None)
        sugar = factory.add_stream("Sugar", [1.0, 0.0], flow_rate=None)
        jam = factory.add_stream("Jam", [0.66667, None], flow_rate=1000, direction="output")
        water = factory.add_stream("Water", [0.0, 1.0], flow_rate=None, direction="output")

        # Ratio: strawberry / sugar = 45/55
        factory.add_ratio("flow", stream1="Strawberry", stream2="Sugar", target_ratio=45 / 55)

        unit = factory.build_process_unit("H-102")

        assert len(unit.ratios) == 1
        assert unit.ratios[0] is not None
        assert unit.is_solvable()

    def test_factory_streams_registry_isolation(self):
        """Test that multiple factories maintain independent stream registries."""
        factory1 = StreamFactory(["Water", "Acetone"], default_flow_type="mole")
        factory2 = StreamFactory(
            ["Ethanol", "Methanol"], default_flow_type="mole"
        )

        stream1 = factory1.add_stream("Stream", [0.5, 0.5], flow_rate=10.0)
        stream2 = factory2.add_stream("Stream", [0.3, 0.7], flow_rate=20.0)

        assert factory1.get_stream("Stream") == stream1
        assert factory2.get_stream("Stream") == stream2
        assert stream1 != stream2
