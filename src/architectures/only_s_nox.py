import torch
import torch.nn as nn
import torch.nn.functional as F

from .node import NODE
from .mlp import MLP
from ..physics_calculator import PhysicsCalculator

# only consider NO2 selectivity, not total NOx

class OnlySelectivityModel(NODE):
    """
Simplified Carbon + NO/NO2 Model: Does not consider NOx sorption equilibria or NOx conversion to N2. Thus only considers NO2 selectivity rather than F_NO2

    Neural Network
        physical state = [m_C, T]
        static = [catalyst_features, inlet_gases, experimental_conditions]

        [z1, z2, z3, z_co2] = NN(static, state)

        y1 = softplus(z1)
        Da2 = softplus(z2)
        y3 = softplus(z3)
        S_CO2 = sigmoid(z_co2)

        Da3 = y3 * m_C

        where:
            y1 controls direct soot oxidation
            Da2 controls the extent of NO/NO2 equilibration
            Da3 controls NO2 consumption by soot
            S_CO2 controls CO2/CO selectivity

        Da2 and Da3 are dimensionless Damkohler-like quantities.

        Since:
            Da3 = y3 * m_C

        y3 has inverse-mass units with respect to the chosen soot-mass units.


    ODE State
        state = [u, T]

        m_C = m_C_initial * exp(-u)

        u(0) = 0

        du/dt = -(dm_C/dt) / m_C
        dT/dt = beta
        Solving this latent state u to ensure numerical stability

        
    Effective Chemical Pathways
        r1      C -> COx
        r2      NO + (0.5 O2) <-> NO2           O2 only considered for the eqbn. NO2
        r3      NO2 + C -> NO + COx


    Direct Soot Oxidation
        r1 = y1 * m_C                           therefore if m_C = 0 then r1 = 0


    NO/NO2 Thermodynamics
        NO + 0.5 O2 <-> NO2

        S_NO2_eq = f(T, O2_in)
        F_NOx_in = F_NO_in + F_NO2_in
        F_NO2_eq = S_NO2_eq * F_NOx_in

            F_NO2_eq is the thermodynamic equilibrium NO2 flow at the current temperature and oxygen concentration.


    Coupled NO2 Formation and Consumption

        Introduce a dimensionless gas-contact coordinate:

            ξ = 0 at reactor inlet
            ξ = 1 at reactor outlet

        Over this effective contact coordinate, NO2 is simultaneously:

            - driven towards NO/NO2 equilibrium
            - consumed through reaction with soot

        The effective NO2 balance is:

            dF_NO2/dξ
                = Da2 * (F_NO2_eq - F_NO2)
                - Da3 * F_NO2

        where:

            Da2 >= 0
                controls approach towards NO/NO2 equilibrium

            Da3 >= 0
                controls NO2 consumption by soot

            Da3 = y3 * m_C


    Derivation of Effective Steady NO2 Flow

        Expanding the balance:

            dF_NO2/dξ
                = Da2 * F_NO2_eq
                - (Da2 + Da3) * F_NO2

        At the effective steady composition:

            dF_NO2/dξ = 0

        therefore:

            Da2 * F_NO2_eq = (Da2 + Da3) * F_NO2_ss

        giving:

            F_NO2_ss = Da2 / (Da2 + Da3) * F_NO2_eq


        Therefore:

            F_NO2_ss = F_NO2_eq              when Da3 = 0

        and:

            F_NO2_ss < F_NO2_eq              when soot consumes NO2


    Derivation of Outlet NO2 Flow

        Define:

            λ = Da2 + Da3

        The NO2 balance becomes:

            dF_NO2/dξ + λ F_NO2
                = Da2 * F_NO2_eq

        whose solution is:

            F_NO2(ξ)
                = F_NO2_ss
                + (F_NO2_in - F_NO2_ss) * exp(-λ ξ)

        Evaluating at the reactor outlet ξ = 1:

            F_NO2_out
                = F_NO2_ss
                + (F_NO2_in - F_NO2_ss)
                  * exp(-(Da2 + Da3))


        This gives a smooth approach towards the kinetically accessible
        NO2 composition rather than applying a hard thermodynamic cap.


    Special Cases

        No soot:
            m_C = 0
            Da3 = 0

            F_NO2_ss = F_NO2_eq

            F_NO2_out
                = F_NO2_eq
                + (F_NO2_in - F_NO2_eq) * exp(-Da2)

            Therefore NO2 smoothly approaches equilibrium.

            Da2 -> 0:
                F_NO2_out -> F_NO2_in

            Da2 -> infinity:
                F_NO2_out -> F_NO2_eq


        No NO/NO2 equilibration:
            Da2 = 0

            F_NO2_ss = 0

            F_NO2_out
                = F_NO2_in * exp(-Da3)

            Therefore only inlet NO2 can oxidise soot.


        No inlet NO2 and no NO oxidation:
            F_NO2_in = 0
            Da2 = 0

            F_NO2_out = 0

            and no NO2-assisted soot oxidation can occur.


    Integrated NO2 Consumption by Soot

        The NO2-assisted carbon oxidation cannot be calculated simply as:

            F_NO2_in - F_NO2_out

        because NO2 can be regenerated from NO through r2 while it is
        simultaneously consumed by soot.

        The local NO2-consumption contribution is:

            dR3/dξ = Da3 * F_NO2(ξ)

        Therefore the total NO2 consumption through the soot pathway is:

            R3 = Da3 * integral_0^1 F_NO2(ξ) dξ

        Using:

            F_NO2(ξ)
                = F_NO2_ss
                + (F_NO2_in - F_NO2_ss) exp(-λξ)

        gives:

            integral_0^1 F_NO2(ξ)dξ
                = F_NO2_ss
                + (F_NO2_in - F_NO2_ss)
                  * (1 - exp(-λ)) / λ

        and therefore:

            R3
                = Da3 * [
                    F_NO2_ss
                    + (F_NO2_in - F_NO2_ss)
                      * (1 - exp(-(Da2 + Da3)))
                      / (Da2 + Da3)
                  ]

        R3 has the same flow units as F_NO2 and represents the amount of
        NO2 consumed by soot over the effective reactor contact.


    Net NO/NO2 Reaction

        The corresponding integrated net NO -> NO2 extent is:

            R2 = integral_0^1
                 Da2 * (F_NO2_eq - F_NO2(ξ)) dξ

        The NO2 balance guarantees:

            F_NO2_out - F_NO2_in = R2 - R3

        Therefore nitrogen is conserved without requiring an explicit
        total-NOx model.


    Nitrogen Outputs

        Assume:
            no NOx adsorption/desorption
            no NOx -> N2 pathway

        Therefore:
            F_NOx_out = F_NOx_in
            F_NO_out
                = F_NOx_in - F_NO2_out
            S_NO2 = F_NO2_out / F_NOx_in

        and automatically:
            F_NO_out + F_NO2_out = F_NOx_in


    Carbon Balance

        Direct oxidation produces:

            r1 = y1 * m_C

        NO2-assisted oxidation produces:

            R3

        Therefore:

            F_COx_out = r1 + R3

        and soot mass changes according to:

            dm_C/dt
                = -carbon_mol_to_mg(F_COx_out) / f_C


    CO / CO2 Outputs

        F_CO2_out
            = S_CO2 * F_COx_out

        F_CO_out
            = (1 - S_CO2) * F_COx_out


    Calculated Experimental Outputs

        soot conversion:
            X = 1 - m_C / m_C_initial

        CO2:
            [CO2] = F_CO2_out / F_total

        CO:
            [CO] = F_CO_out / F_total

        NO2:
            [NO2] = F_NO2_out / F_total

        NO:
            [NO] = F_NO_out / F_total

        NOx:
            [NOx] = F_NOx_in / F_total

        NO2 fraction of NOx:
            S_NO2 = F_NO2_out / F_NOx_in


    Assumptions

        - Total NOx is conserved.
        - NOx adsorption/desorption is ignored.
        - NOx conversion to N2 is ignored.
        - NO2, rather than NO, directly promotes soot oxidation.
        - NO/NO2 equilibrium is treated as a smooth thermodynamic driving
          force rather than a hard upper cap.
        - NO oxidation and NO2 consumption by soot occur simultaneously.
        - Da2 and Da3 are effective Damkohler-like quantities rather than
          separately identifiable rate constants and residence times.
        - Da2 controls the extent of NO/NO2 equilibration.
        - Da3 controls competition between NO2 equilibration and soot
          consumption.
        - Gas-phase NO/NO2 behaviour is quasi-steady relative to the slow
          TPO temperature ramp.
        - Soot mass and temperature remain the only dynamic ODE states.
    """

    def __init__(self, config, nn_input_dim, scaler):
        super().__init__(config, nn_input_dim, scaler)

        self.physics_calculator = PhysicsCalculator()

        nn_cfg = config["nn"].copy()
        nn_cfg["output_dim"] = 4
        self.nn = MLP(input_dim=nn_input_dim, cfg=nn_cfg)
        scales = config["physics"].get("rate_scales", {})
        self.y1_scale = scales.get("k1", 1.0)

        da_scales = config["physics"].get("damkohler_scales", {})
        self.Da2_scale = da_scales.get("Da2", 1.0)
        self.y3_scale = da_scales.get("y3", 1.0)
        self.diagnostic_output_keys = []

    # def get_carbon_fraction(self):
    #     return self.f_C_min + (self.f_C_max - self.f_C_min) * torch.sigmoid(self.z_C)

    def calculate_rates(self, static_inputs_scaled, m_C_unscaled_state, T_unscaled_state, F_total, F_NO_in, F_NO2_in, o2_fraction):
        if static_inputs_scaled.ndim == 1:
            static_inputs_scaled = static_inputs_scaled.unsqueeze(0)

        dtype, device = static_inputs_scaled.dtype, static_inputs_scaled.device

        m_C = self.as_tensor(m_C_unscaled_state, dtype, device)
        T = self.as_tensor(T_unscaled_state, dtype, device)
        F_total = self.as_tensor(F_total, dtype, device)
        F_NO_in = self.as_tensor(F_NO_in, dtype, device)
        F_NO2_in = self.as_tensor(F_NO2_in, dtype, device)
        o2_fraction = self.as_tensor(o2_fraction, dtype, device)

        if m_C.ndim == 0: m_C = m_C.unsqueeze(0)
        if T.ndim == 0: T = T.unsqueeze(0)

        state_scaled = self.scale_state(m_C, T)

        if static_inputs_scaled.shape[0] == 1 and state_scaled.shape[0] > 1:
            static_inputs_scaled = static_inputs_scaled.expand(state_scaled.shape[0], -1)

        nn_inputs = torch.cat([static_inputs_scaled, state_scaled], dim=-1)

        terms = self.calculate_model_terms(static_inputs_scaled, state_scaled, nn_inputs, m_C, T)

        y1 = terms["y1"]
        Da2 = terms["Da2"]
        y3 = terms["y3"]
        S_CO2 = terms["S_CO2"]

        Da3 = y3 * m_C

        F_NOx_in = F_NO_in + F_NO2_in

        # Thermodynamic NO/NO2 equilibrium
        S_NO2_eq = self.physics_calculator.calculate_no2_equilibrium_selectivity(T=T, o2_fraction=o2_fraction)
        F_NO2_eq = S_NO2_eq * F_NOx_in

        # Coupled NO oxidation + NO2 soot consumption
        lam = Da2 + Da3

        # Steady composition of the effective contact-coordinate balance
        F_NO2_ss = torch.where(
            lam > self.eps,
            (Da2 / (lam + self.eps)) * F_NO2_eq,
            F_NO2_eq,
        )

        exp_term = torch.exp(-lam)

        # Analytical solution at xi = 1
        F_NO2_out = F_NO2_ss + (F_NO2_in - F_NO2_ss) * exp_term

        # Integral of F_NO2 over xi from 0 -> 1
        phi = torch.where(
            lam > 1e-8,
            -torch.expm1(-lam) / (lam + self.eps),
            torch.ones_like(lam),
        )

        mean_F_NO2 = F_NO2_ss + (F_NO2_in - F_NO2_ss) * phi

        # Integrated NO2 consumption by soot
        R3 = Da3 * mean_F_NO2

        # Integrated net NO -> NO2 reaction, useful as a diagnostic
        R2 = Da2 * (F_NO2_eq - mean_F_NO2)

        # Direct soot oxidation
        r1 = y1 * m_C

        # Total NOx is conserved
        F_NOx_out = F_NOx_in
        F_NO_out = F_NOx_in - F_NO2_out
        F_N2_out = torch.zeros_like(F_NOx_out)

        S_NO2 = torch.where(
            F_NOx_in > self.eps,
            F_NO2_out / (F_NOx_in + self.eps),
            torch.zeros_like(F_NOx_in),
        )

        # Carbon balance
        F_COx_out = r1 + R3
        F_CO2_out = S_CO2 * F_COx_out
        F_CO_out = (1 - S_CO2) * F_COx_out

        dm_C_dt = -self.physics_calculator.carbon_mol_to_mg(F_COx_out)

        rates = {
            "dm_C_dt": dm_C_dt,

            "y1": y1,
            "Da2": Da2,
            "y3": y3,
            "Da3": Da3,

            "r1": r1,
            "R2": R2,
            "R3": R3,

            "S_CO2": S_CO2,
            "S_NO2": S_NO2,
            "S_NO2_eq": S_NO2_eq,

            "F_NO2_eq": F_NO2_eq,
            "F_NO2_ss": F_NO2_ss,

            "F_CO2_out": F_CO2_out,
            "F_CO_out": F_CO_out,

            "F_NOx_out": F_NOx_out,
            "F_NO2_out": F_NO2_out,
            "F_NO_out": F_NO_out,
            "F_N2_out": F_N2_out,
        }

        rates.update(terms.get("diagnostics", {}))
        return rates
            
    def initial_state(self, batch):
        u0 = torch.zeros_like(batch["m_C_initial"])
        return torch.stack([u0, batch["start_temp_K"]], dim=-1)

    def ode_rhs(self, t, state, batch):
        u, T = state[:, 0], state[:, 1]
        m0 = batch["m_C_initial"]

        has_soot = (m0 > self.eps).to(T.dtype)
        m_C = has_soot * m0 * torch.exp(-u)

        rates = self.calculate_rates(
            static_inputs_scaled=batch["static_inputs_scaled"],
            m_C_unscaled_state=m_C,
            T_unscaled_state=T,
            F_total=batch["F_total"],
            F_NO_in=batch["F_NO_in"],
            F_NO2_in=batch["F_NO2_in"],
            o2_fraction=batch["o2_fraction"],
        )

        du_dt = has_soot * -rates["dm_C_dt"] / (m_C + self.eps)
        dT_dt = batch["ramp_rate_K_min"]

        return torch.stack([du_dt, dT_dt], dim=-1)

    def decode_solution(self, solution, batch):
        indices = batch["observation_indices"].to(solution.device)
        B = batch["m_C_initial"].shape[0]
        batch_indices = torch.arange(B, device=solution.device).unsqueeze(1).expand_as(indices)

        u = solution[indices, batch_indices, 0]
        T = solution[indices, batch_indices, 1]
        m_C = batch["m_C_initial"].unsqueeze(1) * torch.exp(-u)

        return {"u": u, "m_C": m_C, "T": T}
    
    def calculate_rates_at_observations(self, batch, m_C, T):
        B, L = m_C.shape
        n_static = batch["static_inputs_scaled"].shape[-1]

        static_flat = batch["static_inputs_scaled"].unsqueeze(1).expand(B, L, n_static).reshape(B * L, n_static)

        def expand(x):
            return x.unsqueeze(1).expand(B, L).reshape(-1)

        rates_flat = self.calculate_rates(
            static_inputs_scaled=static_flat,
            m_C_unscaled_state=m_C.reshape(-1),
            T_unscaled_state=T.reshape(-1),
            F_total=expand(batch["F_total"]),
            F_NO_in=expand(batch["F_NO_in"]),
            F_NO2_in=expand(batch["F_NO2_in"]),
            o2_fraction=expand(batch["o2_fraction"]),
        )

        rates = {}

        for name, value in rates_flat.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == B * L:
                rates[name] = value.reshape(B, L, *value.shape[1:])
            else:
                rates[name] = value

        return rates
    
    def calculate_outputs_from_trajectory(self, batch, trajectory):
        m_C, T = trajectory["m_C"], trajectory["T"]
        rates = self.calculate_rates_at_observations(batch, m_C, T)

        m0 = batch["m_C_initial"].unsqueeze(1)
        Ft = batch["F_total"].unsqueeze(1)

        soot_conversion = torch.where(
            m0 > self.eps,
            self.physics_calculator.calculate_soot_conversion(remaining_mass=m_C, initial_mass=m0),
            torch.zeros_like(m_C),
        )

        outputs = {
            "temperature_K": T,

            "mass_soot_remaining_mg": m_C,
            "soot_oxidation_conversion": soot_conversion,

            "soot_oxidation_co2_concentration_ppm": self.physics_calculator.mol_min_to_ppm(rates["F_CO2_out"], Ft),
            "soot_oxidation_co_concentration_ppm": self.physics_calculator.mol_min_to_ppm(rates["F_CO_out"], Ft),
            "soot_oxidation_co2_selectivity": rates["S_CO2"],

            "no2_fraction_of_nox": rates["S_NO2"],
            "nox_ppm": self.physics_calculator.mol_min_to_ppm(rates["F_NOx_out"], Ft),
            "no2_ppm": self.physics_calculator.mol_min_to_ppm(rates["F_NO2_out"], Ft),
            "no_ppm": self.physics_calculator.mol_min_to_ppm(rates["F_NO_out"], Ft),
            "n2_ppm": self.physics_calculator.mol_min_to_ppm(rates["F_N2_out"], Ft),

            "S_NO2_eq": rates["S_NO2_eq"],
            "F_NO2_eq": rates["F_NO2_eq"],
            "F_NO2_ss": rates["F_NO2_ss"],

            "Da2": rates["Da2"],
            "Da3": rates["Da3"],
            "r1": rates["r1"],
            "R2": rates["R2"],
            "R3": rates["R3"],
        }
        
        for key in self.diagnostic_output_keys:
            if key in rates:
                outputs[key] = rates[key]

        return outputs
    
    def calculate_model_terms(self, static_inputs_scaled, state_scaled, nn_inputs, m_C, T):
        z1, z2, z3, z_co2 = self.nn(nn_inputs).unbind(dim=-1)

        return {
            "y1": self.y1_scale * F.softplus(z1),
            "Da2": self.Da2_scale * F.softplus(z2),
            "y3": self.y3_scale * F.softplus(z3),
            "S_CO2": torch.sigmoid(z_co2),
        }