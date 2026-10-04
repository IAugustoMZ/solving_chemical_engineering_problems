"""Tabular material-balance reporting."""

import pandas as pd


def build_material_balance_dataframe(streams) -> pd.DataFrame:
    """Build the basis-specific stream results table used by the public API."""
    component_names = sorted(
        {component.name for stream in streams for component in stream.components}
    )
    rows = [("mass_flow_rate", ""), ("molar_flow_rate", "")]
    for variable in (
        "component_mass_flow_rate",
        "component_molar_flow_rate",
        "mass_fraction",
        "mole_fraction",
    ):
        rows.extend((variable, name) for name in component_names)
    table = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(rows, names=("variable", "component"))
    )
    for stream in streams:
        values = {
            ("mass_flow_rate", ""): stream.mass_flow_rate,
            ("molar_flow_rate", ""): stream.molar_flow_rate,
        }
        mass_fractions = stream.mass_fractions
        mole_fractions = stream.mole_fractions
        for component_name in component_names:
            values[("mass_fraction", component_name)] = (
                None if mass_fractions is None else mass_fractions.get(component_name)
            )
            values[("mole_fraction", component_name)] = (
                None if mole_fractions is None else mole_fractions.get(component_name)
            )
            values[("component_mass_flow_rate", component_name)] = (
                None
                if stream.mass_flow_rate is None or mass_fractions is None
                else stream.mass_flow_rate * mass_fractions.get(component_name, 0.0)
            )
            values[("component_molar_flow_rate", component_name)] = (
                None
                if stream.molar_flow_rate is None or mole_fractions is None
                else stream.molar_flow_rate * mole_fractions.get(component_name, 0.0)
            )
        table[stream.name] = pd.Series(values)
    return table
