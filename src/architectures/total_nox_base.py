import torch
import torch.nn as nn

from .node import NODE
from ..physics_calculator import PhysicsCalculator


class SootNOxNeuralODE(NODE):
    def __init__(self, config, nn_input_dim, scaler):
        super().__init__(config, nn_input_dim, scaler)

        self.nn_input_dim = nn_input_dim
        self.physics_calculator = PhysicsCalculator()
        self.f_C_min, self.f_C_max = config["nn"]["oxidisable_mass_fraction_bounds"]
        self.z_C = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        rate_scales = config["physics"].get("rate_scales", {})
        self.k1_scale = rate_scales.get("k1", 1.0)
        self.k2_scale = rate_scales.get("k2", 1.0)
        self.k5_scale = rate_scales.get("k5", 1.0)

        self.diagnostic_output_keys = []

    def get_carbon_fraction(self):
        return self.f_C_min + (self.f_C_max - self.f_C_min) * torch.sigmoid(self.z_C)

    def calculate_model_terms(self, static_inputs_scaled, state_scaled, nn_inputs, m_C, T):
        """
        Must return:
            k1
            k2
            eta3
            z4
            k5
            S_CO2

        May additionally return:
            diagnostics: dict
        """
        raise NotImplementedError

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

        k1, k2, k5 = terms["k1"], terms["k2"], terms["k5"]
        eta3, z4, S_CO2 = terms["eta3"], terms["z4"], terms["S_CO2"]

        F_NOx_in = F_NO_in + F_NO2_in
        nox_mask = (F_NOx_in > 0).to(T.dtype)
        soot_mask = (m_C > 0).to(T.dtype)

        # NO + 0.5 O2 <-> NO2 equilibrium
        S_NO2_eq = self.physics_calculator.calculate_no2_equilibrium_selectivity(T=T, o2_fraction=o2_fraction)
        F_NO2_eq = S_NO2_eq * F_NOx_in

        # Kinetic NO oxidation, capped by thermodynamic equilibrium
        F_NO2_kinetic = F_NO2_in + eta3 * F_NO_in
        F_NO2_potential = torch.minimum(F_NO2_kinetic, F_NO2_eq)

        r3 = nox_mask * (F_NO2_potential - F_NO2_in)

        r1 = soot_mask * k1 * m_C
        r2_raw = nox_mask * soot_mask * k2 * m_C * F_NO2_potential
        r5_raw = nox_mask * soot_mask * k5 * m_C * F_NO2_potential

        no2_demand = r2_raw + r5_raw
        no2_scale = torch.minimum(torch.ones_like(no2_demand), F_NO2_potential / (no2_demand + self.eps))
        r2, r5 = r2_raw * no2_scale, r5_raw * no2_scale

        F_NO2_star = F_NO2_potential - r2 - r5
        F_NOx_star = F_NOx_in - r5

        S_NO2 = torch.where(F_NOx_star > self.eps, F_NO2_star / (F_NOx_star + self.eps), torch.zeros_like(F_NOx_star))

        r4 = nox_mask * F_NOx_star * torch.tanh(z4)

        F_NOx_out = F_NOx_star - r4
        F_NO2_out = S_NO2 * F_NOx_out
        F_NO_out = (1 - S_NO2) * F_NOx_out
        F_N2_out = 0.5 * r5

        F_COx_out = r1 + r2 + r5
        F_CO2_out = S_CO2 * F_COx_out
        F_CO_out = (1 - S_CO2) * F_COx_out

        f_C = self.get_carbon_fraction()
        dm_C_dt = -self.physics_calculator.carbon_mol_to_mg(F_COx_out) / f_C

        rates = {
            "dm_C_dt": dm_C_dt,
            "oxidisable_mass_frac": f_C,

            "r1": r1, "r2": r2, "r3": r3, "r4": r4, "r5": r5,
            "k1": k1, "k2": k2, "k5": k5,

            "S_CO2": S_CO2,
            "S_NO2": S_NO2,
            "S_NO2_eq": S_NO2_eq,

            "F_NO2_eq": F_NO2_eq,
            "F_NO2_potential": F_NO2_potential,

            "F_CO2_out": F_CO2_out,
            "F_CO_out": F_CO_out,

            "F_NOx_out": F_NOx_out,
            "F_NO2_out": F_NO2_out,
            "F_NO_out": F_NO_out,
            "F_N2_out": F_N2_out,

            "CO2_fraction": F_CO2_out / F_total,
            "CO_fraction": F_CO_out / F_total,
            "NOx_fraction": F_NOx_out / F_total,
            "NO2_fraction": F_NO2_out / F_total,
            "NO_fraction": F_NO_out / F_total,
            "N2_fraction": F_N2_out / F_total,
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
        indices = batch["observation_indices"]
        B = batch["m_C_initial"].shape[0]
        batch_indices = torch.arange(B, device=solution.device).unsqueeze(1).expand_as(indices)

        u = solution[indices, batch_indices, 0]
        T = solution[indices, batch_indices, 1]
        m_C = batch["m_C_initial"].unsqueeze(1) * torch.exp(-u)

        return {"u": u, "m_C": m_C, "T": T}

    def calculate_outputs_from_trajectory(self, batch, trajectory):
        rates = self.calculate_rates_at_observations(
            static_inputs_scaled=batch["static_inputs_scaled"],
            m_C=trajectory["m_C"],
            T=trajectory["T"],
            F_total=batch["F_total"],
            F_NO_in=batch["F_NO_in"],
            F_NO2_in=batch["F_NO2_in"],
            o2_fraction=batch["o2_fraction"],
        )

        return self.calculate_outputs(rates, trajectory["m_C"], trajectory["T"], batch["m_C_initial"], batch["F_total"])

    def calculate_rates_at_observations(self, static_inputs_scaled, m_C, T, F_total, F_NO_in, F_NO2_in, o2_fraction):
        B, L = m_C.shape
        n_static = static_inputs_scaled.shape[-1]

        static_flat = static_inputs_scaled.unsqueeze(1).expand(B, L, n_static).reshape(B * L, n_static)

        def expand(x):
            return x.unsqueeze(1).expand(B, L).reshape(-1)

        rates_flat = self.calculate_rates(
            static_inputs_scaled=static_flat,
            m_C_unscaled_state=m_C.reshape(-1),
            T_unscaled_state=T.reshape(-1),
            F_total=expand(F_total),
            F_NO_in=expand(F_NO_in),
            F_NO2_in=expand(F_NO2_in),
            o2_fraction=expand(o2_fraction),
        )

        rates = {}
        for name, value in rates_flat.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == B * L:
                rates[name] = value.reshape(B, L, *value.shape[1:])
            else:
                rates[name] = value

        return rates

    def calculate_outputs(self, rates, m_C, T, m_C_initial, F_total):
        m0 = m_C_initial.unsqueeze(1) if m_C_initial.ndim == 1 and m_C.ndim == 2 else m_C_initial
        Ft = F_total.unsqueeze(1) if F_total.ndim == 1 and m_C.ndim == 2 else F_total

        outputs = {
            "temperature_K": T,
            "mass_soot_remaining_mg": m_C,
            "oxidisable_mass_frac": rates["oxidisable_mass_frac"],
            "soot_oxidation_conversion": self.physics_calculator.calculate_soot_conversion(m_C, m0),

            "soot_oxidation_co2_concentration_ppm": self.physics_calculator.mol_min_to_ppm(rates["F_CO2_out"], Ft),
            "soot_oxidation_co_concentration_ppm": self.physics_calculator.mol_min_to_ppm(rates["F_CO_out"], Ft),
            "soot_oxidation_co2_selectivity": rates["S_CO2"],

            "no2_fraction_of_nox": rates["S_NO2"],
            "nox_ppm": self.physics_calculator.mol_min_to_ppm(rates["F_NOx_out"], Ft),
            "no2_ppm": self.physics_calculator.mol_min_to_ppm(rates["F_NO2_out"], Ft),
            "no_ppm": self.physics_calculator.mol_min_to_ppm(rates["F_NO_out"], Ft),
            "n2_ppm": self.physics_calculator.mol_min_to_ppm(rates["F_N2_out"], Ft),

            "S_NO2_eq": rates["S_NO2_eq"],
            "F_NO2_potential": rates["F_NO2_potential"],

            "r1": rates["r1"], "r2": rates["r2"], "r3": rates["r3"], "r4": rates["r4"], "r5": rates["r5"],
        }

        for key in self.diagnostic_output_keys:
            if key in rates:
                outputs[key] = rates[key]

        return outputs