"""Shared helpers for working with material-flow bases."""


def basis_from_flow_type(flow_type: str) -> str:
    """Map a user-facing flow unit label to a physical flow basis."""
    normalized = str(flow_type).strip().lower()
    if any(token in normalized for token in ("mole", "molar", "mol", "kmol")):
        return "molar"
    if any(token in normalized for token in ("mass", "kg", "gram", " g", "ton")):
        return "mass"
    if "volume" in normalized or normalized in ("volumetric", "m3/h", "l/min"):
        return "volume"
    raise ValueError(
        f"Unsupported flow_type '{flow_type}': use mass, molar, or legacy volumetric units."
    )
