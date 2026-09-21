"""
MaterialBalanceSystem — general multi-stream, multi-component steady-state
material balance analyser and linear solver.

Design principles
-----------------
Single Responsibility:
    Each method has one clearly defined task (preprocess, count unknowns,
    build the linear system, etc.).
Open/Closed:
    The DOF counting and linear system construction are separated so that
    subclasses can extend to reactive or unsteady systems without touching
    the solver core.
Dependency Inversion:
    Validation logic is imported from ``src.utils.validators`` rather than
    duplicated here.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from src.material_balances.models.stream import Stream
from src.utils.validators import check_fraction, check_fractions_sum


class MaterialBalanceSystem:
    """
    Steady-state, non-reactive multi-stream, multi-component material balance system.

    Given a set of input and output :class:`Stream` objects and a list of component
    names, this class:

    1. **Preprocesses** streams: validates known fractions and resolves fractions
       that are uniquely determined by the sum-to-one constraint.
    2. **Analyses** the degree of freedom (DOF) of the system.
    3. **Solves** the resulting linear system when DOF == 0.
    4. **Reports** results in a structured dictionary and as a formatted string.

    Degree-of-freedom formula (non-reactive, steady-state):
        - Independent equations = C  (one component balance per component)
        - Independent unknowns per stream = 1 (flow) + max(0, C−1 − k_known)
          where k_known is the number of known fractions in that stream.
        - DOF = total unknowns − total equations.

    The sum-to-one constraint reduces independent fraction unknowns per stream
    from C to C−1 (one fraction is always dependent).

    Args:
        components (List[str]): Ordered list of component names present in the system
            (e.g., ``['A', 'B', 'C']``).
        streams (List[Stream]): All input and output :class:`Stream` objects.

    Raises:
        ValueError: If any stream does not have a fraction entry (even if None)
            for every component in ``components``.

    Example:
        >>> system = MaterialBalanceSystem(
        ...     components=['A', 'B', 'C'],
        ...     streams=[
        ...         Stream('S1', 'input',  None,   {'A': 0.0,  'B': 0.03, 'C': 0.97}),
        ...         Stream('S2', 'input',  5300.0, {'A': None, 'B': None, 'C': None}),
        ...         Stream('S3', 'output', None,   {'A': 1.0,  'B': 0.0,  'C': 0.0}),
        ...         Stream('S4', 'output', 1200.0, {'A': 0.70, 'B': None, 'C': None}),
        ...         Stream('S5', 'output', None,   {'A': 0.0,  'B': 0.60, 'C': 0.40}),
        ...     ],
        ... )
        >>> result = system.solve_or_analyze()
    """

    def __init__(self, components: List[str], streams: List[Stream]) -> None:
        self.components: List[str] = components
        self.streams: List[Stream] = streams
        self._preprocessed: bool = False
        self._validate_component_coverage()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_component_coverage(self) -> None:
        """
        Ensures every stream has a fraction entry for every system component.

        Missing entries could be silently treated as zero, which would introduce
        hard-to-diagnose errors.  We require explicit values (float or None) for
        every (stream, component) pair.

        Raises:
            ValueError: If any component is absent from any stream's fractions dict.
        """
        for s in self.streams:
            missing = [c for c in self.components if c not in s.fractions]
            if missing:
                raise ValueError(
                    f"Stream '{s.name}' is missing fraction entries for component(s): "
                    f"{missing}. Specify the value (float) or None for each component."
                )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self) -> None:
        """
        Validates known fractions and resolves deterministic unknowns.

        For each stream:

        - Every known fraction is validated to be in [0, 1].
        - If exactly **one** fraction is None, it is computed from the
          sum-to-one constraint and stored back into the stream in-place.
        - If **all** fractions are known, their sum is verified to equal 1.

        This step runs before DOF analysis and solving.  It is idempotent;
        calling it more than once is safe.

        Raises:
            ValueError: If a known fraction is outside [0, 1], if known fractions
                already exceed 1 (making the remaining fraction negative), or if
                all fractions are known but do not sum to 1.
        """
        for s in self.streams:
            known = {c: v for c, v in s.fractions.items() if v is not None}
            unknown_keys = [c for c in self.components if s.fractions.get(c) is None]

            # Validate all known fractions
            for comp, val in known.items():
                check_fraction(val, name=f"Fraction of '{comp}' in stream '{s.name}'")

            if len(unknown_keys) == 1:
                # Uniquely determined: compute from sum constraint
                determined = 1.0 - sum(known.values())
                if determined < -1e-6:
                    raise ValueError(
                        f"Stream '{s.name}': known fractions already sum to "
                        f"{sum(known.values()):.6f}, which exceeds 1."
                    )
                s.fractions[unknown_keys[0]] = max(0.0, determined)

            elif len(unknown_keys) == 0:
                # All known: verify sum = 1
                check_fractions_sum(
                    [s.fractions[c] for c in self.components],
                    name=f"fractions in stream '{s.name}'"
                )

        self._preprocessed = True

    # ------------------------------------------------------------------
    # DOF analysis
    # ------------------------------------------------------------------

    def count_unknowns(self) -> int:
        """
        Counts independent unknowns in the system.

        For each stream the contribution is:

        .. code-block:: text

            +1                            if total_flow is None
            +max(0, (C - 1) - k_known)   for fraction unknowns

        The ``(C - 1)`` factor reflects the sum-to-one constraint: one fraction
        per stream is always dependent on the others and is therefore **not** an
        independent unknown.

        Returns:
            int: Total number of independent unknowns.
        """
        total = 0
        C = len(self.components)
        for s in self.streams:
            if s.total_flow is None:
                total += 1
            k_known = s.n_known_fractions(self.components)
            total += max(0, (C - 1) - k_known)
        return total

    def count_independent_equations(self) -> int:
        """
        Returns the number of independent material balance equations.

        For a **non-reactive**, **steady-state** system:

        - One balance can be written per component (C independent equations).
        - The total (overall) mass balance is the algebraic sum of all component
          balances and is therefore **not** independent.

        Returns:
            int: Number of independent equations (equals the number of components).
        """
        return len(self.components)

    def degree_of_freedom(self) -> int:
        """
        Computes the degree of freedom (DOF) of the system.

        .. math::

            \\text{DOF} = N_{\\text{unknowns}} - N_{\\text{equations}}

        Interpretation:

        - **DOF = 0** → system is exactly determined; call :meth:`solve`.
        - **DOF > 0** → under-determined; DOF more pieces of information are needed.
        - **DOF < 0** → over-determined; the system may be inconsistent.

        Returns:
            int: Degree of freedom.
        """
        return self.count_unknowns() - self.count_independent_equations()

    def list_unknowns(self) -> List[str]:
        """
        Returns a human-readable description of every current independent unknown.

        For fraction unknowns, only the ``(C - 1)`` independent ones are listed
        per stream (one fraction is always determined by the sum constraint).

        Returns:
            List[str]: One string per independent unknown variable.
        """
        result: List[str] = []
        C = len(self.components)
        for s in self.streams:
            if s.total_flow is None:
                result.append(f"Total flow of stream '{s.name}'")
            k_known = s.n_known_fractions(self.components)
            n_free = max(0, (C - 1) - k_known)
            unknown_fracs = [c for c in self.components if s.fractions.get(c) is None]
            # Report only the n_free independent ones (last is sum-determined)
            for comp in unknown_fracs[:n_free]:
                result.append(f"Fraction of '{comp}' in stream '{s.name}'")
        return result

    def missing_info_suggestions(self) -> List[str]:
        """
        Generates targeted suggestions for what to specify when DOF > 0.

        Each suggestion corresponds to one independent unknown that must be
        supplied to reduce the DOF by one.

        Returns:
            List[str]: Suggestion strings, one per needed variable.
                       Empty if the system is already determined (DOF ≤ 0).
        """
        dof = self.degree_of_freedom()
        if dof <= 0:
            return []
        return [f"→ Specify: {u}" for u in self.list_unknowns()]

    # ------------------------------------------------------------------
    # Internal solver helpers
    # ------------------------------------------------------------------

    def _get_raw_unknowns(self) -> List[Tuple[str, str]]:
        """
        Collects ALL None values across streams as ``(stream_name, variable)`` tuples.

        Variable is ``'flow'`` for the total flow rate, or a component name for fractions.
        This list drives the column ordering of the linear system matrix.

        Returns:
            List[Tuple[str, str]]: One tuple per raw unknown (after preprocessing).
        """
        unknowns: List[Tuple[str, str]] = []
        for s in self.streams:
            if s.total_flow is None:
                unknowns.append((s.name, 'flow'))
            for comp in self.components:
                if s.fractions.get(comp) is None:
                    unknowns.append((s.name, comp))
        return unknowns

    def _check_linearity(self) -> bool:
        """
        Checks whether the balance system is linear.

        A system is **nonlinear** when a stream has *both* its total flow rate
        *and* at least one fraction unknown, because the product
        ``flow × fraction`` would be a product of two unknowns.

        Returns:
            bool: True if linear (solvable by :mod:`numpy.linalg`), False otherwise.
        """
        for s in self.streams:
            if s.total_flow is None:
                if any(s.fractions.get(c) is None for c in self.components):
                    return False
        return True

    def _build_linear_system(
        self,
        unknowns: List[Tuple[str, str]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs the augmented coefficient matrix **A** and RHS vector **b**
        for the system ``A @ x = b``.

        **Row structure**

        1. *Component balance rows* (C rows): one per component, expressing
           ``Σ_inputs (F_i · x_{i,c}) - Σ_outputs (F_j · x_{j,c}) = 0``.
           Known terms are moved to **b**; unknown terms contribute to **A**.

        2. *Sum-constraint rows*: for each stream with ≥ 2 unknown fractions,
           ``Σ_c x_{s,c} = 1`` provides an additional independent equation.

        Args:
            unknowns: Ordered list of ``(stream_name, variable)`` tuples that
                define the column layout of **A**.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The (A, b) pair.

        Raises:
            ValueError: If a stream has both its flow rate and a fraction unknown
                (nonlinear case — not supported by this solver).
        """
        n = len(unknowns)
        # Build an index map for O(1) column lookups
        uk_index: Dict[Tuple[str, str], int] = {u: i for i, u in enumerate(unknowns)}

        A_rows: List[np.ndarray] = []
        b_rows: List[float] = []

        # ---- Component balance equations ----
        for comp in self.components:
            row = np.zeros(n)
            rhs = 0.0

            for s in self.streams:
                F = s.total_flow
                x = s.fractions.get(comp)

                if F is not None and x is not None:
                    # Both known → move to RHS (sign: input adds, output subtracts)
                    rhs -= s.sign * F * x

                elif F is None and x is not None:
                    # Flow unknown → coefficient is the known fraction
                    row[uk_index[(s.name, 'flow')]] += s.sign * x

                elif F is not None and x is None:
                    # Fraction unknown → coefficient is the known flow
                    row[uk_index[(s.name, comp)]] += s.sign * F

                else:
                    raise ValueError(
                        f"Nonlinear system detected: both total flow and fraction of "
                        f"'{comp}' are unknown in stream '{s.name}'. "
                        "Nonlinear solving is not supported."
                    )

            A_rows.append(row)
            b_rows.append(rhs)

        # ---- Fraction sum-constraint equations ----
        # For each stream with ≥2 unknown fractions, Σ x_c = 1 is an extra equation.
        for s in self.streams:
            unknown_fracs = [c for c in self.components if s.fractions.get(c) is None]
            if len(unknown_fracs) >= 2:
                row = np.zeros(n)
                known_sum = sum(
                    s.fractions[c] for c in self.components
                    if s.fractions.get(c) is not None
                )
                rhs = 1.0 - known_sum
                for comp in unknown_fracs:
                    row[uk_index[(s.name, comp)]] = 1.0
                A_rows.append(row)
                b_rows.append(rhs)

        return np.array(A_rows), np.array(b_rows)

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    def solve(self) -> Dict[Tuple[str, str], float]:
        """
        Solves the material balance system as a linear system of equations.

        Requires DOF == 0.  Builds the augmented system (component balances +
        sum constraints) and solves it with :func:`numpy.linalg.solve`.

        Returns:
            Dict[Tuple[str, str], float]: Mapping from ``(stream_name, variable)``
            to the computed value.  ``variable`` is ``'flow'`` or a component name.

        Raises:
            ValueError: If DOF ≠ 0 or the system is nonlinear.
            numpy.linalg.LinAlgError: If the coefficient matrix is singular.
        """
        if not self._preprocessed:
            self.preprocess()

        dof = self.degree_of_freedom()
        if dof != 0:
            msg = "Provide more information." if dof > 0 else "System is over-determined."
            raise ValueError(f"Cannot solve: DOF = {dof}. {msg}")

        if not self._check_linearity():
            raise ValueError(
                "Nonlinear system detected (a stream has both its flow rate and "
                "a fraction unknown). Nonlinear solving is not yet supported."
            )

        unknowns = self._get_raw_unknowns()
        A, b = self._build_linear_system(unknowns)

        if A.shape[0] != A.shape[1]:
            raise AssertionError(
                f"Internal error: system matrix is not square ({A.shape}). "
                "Please report this bug."
            )

        try:
            x_sol = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "Failed to solve (singular matrix). Verify that the system is truly "
                f"determined and that no two equations are linearly dependent. Detail: {exc}"
            ) from exc

        return {unknowns[i]: float(x_sol[i]) for i in range(len(unknowns))}

    def _apply_solution(
        self, solution: Dict[Tuple[str, str], float]
    ) -> None:
        """
        Writes solved values back into the stream objects in-place.

        After applying the direct solution, any fraction that is still None
        (because it was the sum-determined one and not included as a solver
        unknown) is computed from the residual of the sum constraint.

        Args:
            solution: Dict returned by :meth:`solve`.
        """
        # Apply directly solved values
        for s in self.streams:
            if (s.name, 'flow') in solution:
                s.total_flow = solution[(s.name, 'flow')]
            for comp in self.components:
                if (s.name, comp) in solution:
                    s.fractions[comp] = solution[(s.name, comp)]

        # Fill any remaining None fractions via sum constraint
        for s in self.streams:
            still_none = [c for c in self.components if s.fractions.get(c) is None]
            if len(still_none) == 1:
                known_sum = sum(
                    s.fractions[c] for c in self.components
                    if s.fractions.get(c) is not None
                )
                s.fractions[still_none[0]] = max(0.0, 1.0 - known_sum)

    def _validate_solution(self) -> List[str]:
        """
        Post-solve sanity check on the fully populated stream data.

        Verifies that all fractions lie in [0, 1], that fractions sum to 1 for
        each stream, and that the component balances are satisfied to within a
        relative tolerance of 1 × 10⁻⁶.

        Returns:
            List[str]: Warning strings for any violated check (empty = no issues).
        """
        warnings: List[str] = []
        for s in self.streams:
            for comp in self.components:
                val = s.fractions.get(comp)
                if val is None:
                    continue
                if val < -1e-6 or val > 1.0 + 1e-6:
                    warnings.append(
                        f"[WARNING] Stream '{s.name}': fraction of '{comp}' = "
                        f"{val:.6f} is outside [0, 1]."
                    )
            frac_sum = sum(
                v for v in s.fractions.values() if v is not None
            )
            if abs(frac_sum - 1.0) > 1e-5:
                warnings.append(
                    f"[WARNING] Stream '{s.name}': fractions sum to {frac_sum:.6f} (expected 1)."
                )

        # Component balance residuals
        for comp in self.components:
            residual = sum(
                s.sign * (s.total_flow or 0.0) * (s.fractions.get(comp) or 0.0)
                for s in self.streams
            )
            scale = max(
                abs((s.total_flow or 0.0) * (s.fractions.get(comp) or 0.0))
                for s in self.streams
            ) or 1.0
            if abs(residual / scale) > 1e-4:
                warnings.append(
                    f"[WARNING] Component '{comp}' balance not satisfied "
                    f"(residual = {residual:.4f})."
                )

        return warnings

    # ------------------------------------------------------------------
    # Result assembly & reporting
    # ------------------------------------------------------------------

    def to_result_dict(
        self,
        override_dof: Optional[int] = None,
        override_unknowns: Optional[int] = None
    ) -> dict:
        """
        Assembles a structured result dictionary from the current state of all streams.

        The dictionary has the following top-level keys:

        - ``'analysis'``: DOF metadata (components, stream counts, DOF value, status).
        - ``'streams'``: Per-stream results (uses :meth:`Stream.to_report_dict`).
        - ``'suggestions'``: Present only if DOF > 0; human-readable suggestions for
          what information to provide.
        - ``'warnings'``: List of post-solve validation warnings (present only when
          the system was solved).

        Args:
            override_dof: If provided, uses this DOF instead of recalculating.
                Useful after solving, since solving populates unknowns and would
                otherwise result in a negative calculated DOF.
            override_unknowns: If provided, uses this instead of recalculating.

        Returns:
            dict: Fully populated result dictionary.
        """
        input_streams = [s for s in self.streams if s.is_input]
        output_streams = [s for s in self.streams if s.is_output]
        dof = self.degree_of_freedom() if override_dof is None else override_dof
        n_unknowns = self.count_unknowns() if override_unknowns is None else override_unknowns

        analysis = {
            'components': self.components,
            'n_streams': len(self.streams),
            'n_inputs': len(input_streams),
            'n_outputs': len(output_streams),
            'n_independent_balances': self.count_independent_equations(),
            'n_unknowns': n_unknowns,
            'degree_of_freedom': dof,
            'status': (
                'solved' if dof == 0
                else 'underdetermined' if dof > 0
                else 'overdetermined'
            ),
        }

        streams_data = {
            s.name: s.to_report_dict(self.components)
            for s in self.streams
        }

        result: dict = {'analysis': analysis, 'streams': streams_data}

        if dof > 0:
            result['suggestions'] = self.missing_info_suggestions()

        return result

    def print_report(self, result: Optional[dict] = None) -> str:
        """
        Prints a human-readable material balance report and returns the string.

        Args:
            result (Optional[dict]): A pre-assembled result dict (e.g., from
                :meth:`to_result_dict`).  If None, assembles it from the current
                stream state.

        Returns:
            str: The formatted report string (also printed to stdout).
        """
        if result is None:
            result = self.to_result_dict()

        a = result['analysis']
        W = 62
        lines = [
            "=" * W,
            "  MATERIAL BALANCE SYSTEM REPORT",
            "=" * W,
            f"  Components    : {', '.join(a['components'])}",
            f"  Input streams : {a['n_inputs']}",
            f"  Output streams: {a['n_outputs']}",
            "-" * W,
            "  DEGREE OF FREEDOM ANALYSIS",
            "-" * W,
            f"  {'How many independent material balances can be written?':<50} {a['n_independent_balances']}",
            f"  {'Total independent unknowns:':<50} {a['n_unknowns']}",
            f"  {'Degree of Freedom (DOF = unknowns - equations):':<50} {a['degree_of_freedom']}",
        ]

        dof = a['degree_of_freedom']
        if dof == 0:
            lines.append(
                f"  {'Status:':<50} ✓ Exactly determined — proceeding to solve."
            )
        elif dof > 0:
            lines.append(
                f"  {'Status:':<50} ✗ Under-determined."
            )
            lines.append(
                f"\n  {dof} more piece(s) of information must be specified."
            )
            lines.append("  Suggestions:")
            for sug in result.get('suggestions', []):
                lines.append(f"    {sug}")
        else:
            lines.append(
                f"  {'Status:':<50} ✗ Over-determined (may be inconsistent)."
            )

        # Stream results (only if solved)
        if a['status'] == 'solved':
            lines += ["-" * W, "  STREAM RESULTS", "-" * W]
            for sname, data in result['streams'].items():
                label = f"[{data['direction'].upper():6}]  {sname}"
                lines.append(f"\n  {label}")
                flow = data['total_flow']
                lines.append(f"    {'Total flow':>20} : {flow:>12.4f}")
                lines.append(f"    {'Component':>20}   {'Fraction':>10}   {'Component flow':>14}")
                lines.append(f"    {'-'*20}   {'-'*10}   {'-'*14}")
                for comp in a['components']:
                    frac = data['fractions'][comp]
                    cf = data['component_flows'][comp]
                    frac_str = f"{frac:>10.4f}" if frac is not None else f"{'N/A':>10}"
                    cf_str = f"{cf:>14.4f}" if cf is not None else f"{'N/A':>14}"
                    lines.append(f"    {comp:>20}   {frac_str}   {cf_str}")

            # Validation warnings
            for w in result.get('warnings', []):
                lines.append(f"\n  {w}")

        lines.append("\n" + "=" * W)
        report = "\n".join(lines)
        print(report)
        return report

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def solve_or_analyze(self, print_report: bool = False) -> dict:
        """
        Full analysis pipeline: preprocess → DOF analysis → solve (if possible).

        If DOF == 0, the system is solved and solved values are written back to
        the stream objects.  The returned dictionary reflects the solved state.

        If DOF ≠ 0, the system is analysed only, and the returned dictionary
        includes suggestions for what information must still be provided.

        Args:
            print_report (bool): If True, also prints the formatted report to stdout.

        Returns:
            dict: Result dictionary from :meth:`to_result_dict`, populated with
                  solved stream values (or suggestions) as appropriate.
        """
        if not self._preprocessed:
            self.preprocess()

        n_unknowns = self.count_unknowns()
        dof = self.degree_of_freedom()
        
        if dof == 0:
            solution = self.solve()
            self._apply_solution(solution)
            result = self.to_result_dict(override_dof=dof, override_unknowns=n_unknowns)
            result['warnings'] = self._validate_solution()
        else:
            result = self.to_result_dict(override_dof=dof, override_unknowns=n_unknowns)

        if print_report:
            self.print_report(result)

        return result

