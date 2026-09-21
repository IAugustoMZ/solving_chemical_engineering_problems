import numpy as np
from typing import List, Optional
from src.utils.validators import check_fraction, check_fractions_sum
from src.material_balances.models.stream import Stream
from src.material_balances.models.balance_system import MaterialBalanceSystem

class SimpleMaterialBalancer:
    def __init__(self):
        """
        class that implements simple material balances
        """

    def time_to_fill_liquid_tank(
            rho: float, 
            volume_list: list, 
            mass_input: float, 
            mass_output: float
        ) -> float:
        """
        calculates the time needed to fill a liquid tank using the general integral material balance

        Args:
            rho (float): fluid's density (in kg/m3)
            volume_list (list): list of volumes measured [initial, final], in m3
            mass_input (float): output mass flow rate in kg/s
            mass_output (float): output mass flow rate, in kg/s

        Returns:
            float: time demanded to fill the tank, in seconds
        """
        deltaV = volume_list[-1] - volume_list[0]
        delta_tank_mass = rho * deltaV
        rate_accumulation = mass_input - mass_output

        # log variables
        print('Initial tank volume (m3): %.2f'%volume_list[0])
        print('Full tank volume (m3): %.2f'%volume_list[-1])
        print('Delta volume in tank (m3): %.2f'%volume_list[0])
        print('Rate of mass input (kg/s): %.2f'%mass_input)
        print('Rate of mass output (kg/s): %.2f'%mass_output)

        if rate_accumulation == 0:
            raise Exception("Zero division, infinite time to fill the tank, because output is equal to input")
        elif rate_accumulation < 0:
            raise Exception("Output is greater than input, the tank will never fill")
        else:
            return delta_tank_mass / rate_accumulation

    def binary_mixture_separation_vl(
            input_molar_flow: float,
            x_1: float,
            k: float,
            z: list
    ) -> dict:
        """
        calculates the material balance of a binary mixture being fed to a separation system that produces a vapor and a liquid streams. In stationary state, no accumulation and V = k.F, where 0 <= k <= 1

        Args:
            input_molar_flow (float): total mixture molar input flow rate (mol/min)
            x_1 (float): molar fraction of component 1 at the liquid stream
            k (float): fraction of input molar flow that goes to vapor output stream
            z (list): list of molar fractions of the inputted mixture (comp1, comp2)

        Returns:
            dict: dictionary with the system's metada: total input molar flow, vapor output stream info and liquid output stream
        """
        # checks
        check_fraction(k, name="The vapor fraction in steady state")
        check_fraction(x_1, name="The liquid molar fraction of component 1 in steady state")
        check_fractions_sum(z, name="molar fractions of the components in input stream")

        def _build_stream_dict(flow, fractions):
            return {
                'molar_flow_mol_min': float(flow),
                'molar_fraction_list': [float(f) for f in fractions],
                'molar_flow_by_component': [float(f * flow) for f in fractions]
            }

        result = {
            'input': _build_stream_dict(input_molar_flow, z)
        }

        # building matrices
        A = np.array([
            [(-k*input_molar_flow), (1-x_1)],
            [k*input_molar_flow, x_1]
        ])
        b = np.array([
            (z[-1] - k)*input_molar_flow,
            z[0]*input_molar_flow
        ])

        # solving the linear system
        x = np.linalg.solve(A, b)
        y_1 = float(x[0])

        # build output metadata
        vapor_flow = k * input_molar_flow
        liquid_flow = (1 - k) * input_molar_flow

        result['vapor'] = _build_stream_dict(vapor_flow, [y_1, 1 - y_1])
        result['liquid'] = _build_stream_dict(liquid_flow, [x_1, 1 - x_1])

        return result

    def multistream_balance(
        components: List[str],
        streams: List[dict],
        print_report: bool = True,
    ) -> dict:
        """
        General multi-component, multi-stream steady-state material balance.

        Accepts any number of input and output streams, each characterised by a
        total flow rate and per-component fractions (molar or mass).  The method:

        1. **Validates** all supplied fractions (range and sum checks).
        2. **Resolves** any fraction that is uniquely determined by the sum-to-one
           constraint (e.g., if only one fraction in a stream is unknown).
        3. **Analyses** the degree of freedom (DOF):

           - Reports how many independent material balances can be written.
           - Reports how many unknowns remain after preprocessing.
           - Reports the DOF = unknowns − equations.

        4. **Solves** the resulting linear system when DOF == 0.
        5. **Reports** the results, optionally printing a formatted summary.

        Assumptions
        -----------
        - **Steady state**: no accumulation inside the system boundary.
        - **No chemical reaction**: no generation or consumption terms.
        - **Linear system**: no stream has *both* its total flow rate *and* any
          fraction unknown simultaneously (that would make the product nonlinear).

        Args:
            components (List[str]): Ordered list of component identifiers
                (e.g., ``['A', 'B', 'C']``).
            streams (List[dict]): One dict per stream. Required keys:

                - ``'name'`` (str): Stream label.
                - ``'direction'`` (str): ``'input'`` or ``'output'``.
                - ``'total_flow'`` (float | None): Total flow rate in the chosen
                  units; None if unknown.
                - ``'fractions'`` (dict[str, float | None]): Fraction of each
                  component; None if unknown. **Every component must have an entry**
                  (use ``0.0`` for truly absent components).

            print_report (bool): If True (default), prints a formatted report to
                stdout after analysis / solving.

        Returns:
            dict: Structured result with the following keys:

            ``'analysis'``:
                Metadata dict with keys ``components``, ``n_inputs``,
                ``n_outputs``, ``n_independent_balances``, ``n_unknowns``,
                ``degree_of_freedom``, and ``status``
                (``'solved'`` | ``'underdetermined'`` | ``'overdetermined'``).

            ``'streams'``:
                Dict mapping each stream name to its result dict:
                ``direction``, ``total_flow``, ``fractions``, ``component_flows``.
                Solved values are populated when status is ``'solved'``;
                unknowns remain ``None`` otherwise.

            ``'suggestions'``:
                *(Present only when DOF > 0)* List of human-readable suggestions
                for what information still needs to be specified.

            ``'warnings'``:
                *(Present only when solved)* List of post-solve sanity-check
                warnings (empty list = no issues detected).

        Raises:
            ValueError: If a stream is missing a fraction entry for any component,
                if a fraction is outside [0, 1], or if fractions do not sum to 1
                when all are known.

        Example — image problem (DOF analysis only, system is under-determined):

            >>> result = SimpleMaterialBalancer.multistream_balance(
            ...     components=['A', 'B', 'C'],
            ...     streams=[
            ...         {'name': 'S1', 'direction': 'input',  'total_flow': None,
            ...          'fractions': {'A': 0.0,  'B': 0.03, 'C': 0.97}},
            ...         {'name': 'S2', 'direction': 'input',  'total_flow': 5300.0,
            ...          'fractions': {'A': None, 'B': None, 'C': None}},
            ...         {'name': 'S3', 'direction': 'output', 'total_flow': None,
            ...          'fractions': {'A': 1.0,  'B': 0.0,  'C': 0.0}},
            ...         {'name': 'S4', 'direction': 'output', 'total_flow': 1200.0,
            ...          'fractions': {'A': 0.70, 'B': None, 'C': None}},
            ...         {'name': 'S5', 'direction': 'output', 'total_flow': None,
            ...          'fractions': {'A': 0.0,  'B': 0.60, 'C': 0.40}},
            ...     ],
            ... )
        """
        stream_objects = [
            Stream(
                name=s['name'],
                direction=s['direction'],
                total_flow=s.get('total_flow'),
                fractions=dict(s.get('fractions', {})),
            )
            for s in streams
        ]

        system = MaterialBalanceSystem(components=components, streams=stream_objects)
        result = system.solve_or_analyze(print_report=print_report)
        return result