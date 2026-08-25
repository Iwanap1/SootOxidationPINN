import torch


class PhysicsCalculator:
    R = 8.314                       # J mol-1 K-1
    CARBON_RMM = 12.011             # g mol-1
    MOLAR_VOLUME_STP_ML = 22414.0   # ml / mol
    K = 273.15                      # C --> K

    # SHOMATE_NO = {
    #     "A": 23.83491,
    #     "B": 12.58878,
    #     "C": -1.139011,
    #     "D": -1.497459,
    #     "E": 0.214194,
    #     "F": 83.35783,
    #     "G": 237.1219,
    #     "H": 90.29114,
    # }

    # SHOMATE_NO2 = {
    #     "A": 16.10857,
    #     "B": 75.89525,
    #     "C": -54.38740,
    #     "D": 14.30777,
    #     "E": 0.239423,
    #     "F": 26.17464,
    #     "G": 240.5386,
    #     "H": 33.09502,
    # }

    # SHOMATE_O2_LOW = {
    #     "A": 31.32234,
    #     "B": -20.23531,
    #     "C": 57.86644,
    #     "D": -36.50624,
    #     "E": -0.007374,
    #     "F": -8.903471,
    #     "G": 246.7945,
    #     "H": 0.0,
    # }

    # SHOMATE_O2_HIGH = {
    #     "A": 30.03235,
    #     "B": 8.772972,
    #     "C": -3.988133,
    #     "D": 0.788313,
    #     "E": -0.741599,
    #     "F": -11.32468,
    #     "G": 236.1663,
    #     "H": 0.0,
    # }

    DELTA_H_NO2_DECOMP = 114.14e3   # J mol-1, 2NO2 -> 2NO + O2
    DELTA_S_NO2_DECOMP = 146.55     # J mol-1 K-1
    P_STANDARD_BAR = 1.0


    def gas_flow_to_mol_min(self, flow_ml_min: torch.Tensor) -> torch.Tensor:
        return flow_ml_min / self.MOLAR_VOLUME_STP_ML

    def ppm_to_mol_min(self, ppm: torch.Tensor, total_flow_mol_min: torch.Tensor) -> torch.Tensor:
        return ppm * 1e-6 * total_flow_mol_min

    def fraction_to_mol_min(self, fraction: torch.Tensor, total_flow_mol_min: torch.Tensor) -> torch.Tensor:
        return fraction * total_flow_mol_min

    def mol_min_to_ppm(self, species_flow: torch.Tensor, total_flow: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return species_flow / (total_flow + eps) * 1e6

    def mol_min_to_fraction(self, species_flow: torch.Tensor, total_flow: torch.Tensor, eps: float = 1e-12,) -> torch.Tensor:
        return species_flow / (total_flow + eps)

    def carbon_mol_to_mg(self, carbon_mol: torch.Tensor) -> torch.Tensor:
        return carbon_mol * self.CARBON_RMM * 1e3

    def calculate_soot_conversion(self, remaining_mass: torch.Tensor, initial_mass: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return 1 - remaining_mass / (initial_mass + eps)

    # def calculate_shomate_H(self, T: torch.Tensor, coeffs: dict) -> torch.Tensor:
    #     t = T / 1000

    #     H_increment = (
    #         coeffs["A"] * t
    #         + coeffs["B"] * t**2 / 2
    #         + coeffs["C"] * t**3 / 3
    #         + coeffs["D"] * t**4 / 4
    #         - coeffs["E"] / t
    #         + coeffs["F"]
    #         - coeffs["H"]
    #     )

    #     return coeffs["H"] + H_increment

    # def calculate_shomate_S(self, T: torch.Tensor, coeffs: dict) -> torch.Tensor:
    #     t = T / 1000

    #     return (
    #         coeffs["A"] * torch.log(t)
    #         + coeffs["B"] * t
    #         + coeffs["C"] * t**2 / 2
    #         + coeffs["D"] * t**3 / 3
    #         - coeffs["E"] / (2 * t**2)
    #         + coeffs["G"]
    #     )

    # def calculate_species_H_S(self, T: torch.Tensor, species: str):
    #     if species == "NO":
    #         coeffs = self.SHOMATE_NO

    #     elif species == "NO2":
    #         coeffs = self.SHOMATE_NO2

    #     elif species == "O2":
    #         H_low = self.calculate_shomate_H(T, self.SHOMATE_O2_LOW)
    #         S_low = self.calculate_shomate_S(T, self.SHOMATE_O2_LOW)

    #         H_high = self.calculate_shomate_H(T, self.SHOMATE_O2_HIGH)
    #         S_high = self.calculate_shomate_S(T, self.SHOMATE_O2_HIGH)

    #         H = torch.where(T < 700, H_low, H_high)
    #         S = torch.where(T < 700, S_low, S_high)

    #         return H, S

    #     else:
    #         raise ValueError(f"Unknown species: {species}")

    #     H = self.calculate_shomate_H(T, coeffs)
    #     S = self.calculate_shomate_S(T, coeffs)

    #     return H, S

    # def calculate_no2_equilibrium_constant(self, T: torch.Tensor) -> torch.Tensor:
    #     H_NO, S_NO = self.calculate_species_H_S(T, "NO")
    #     H_NO2, S_NO2 = self.calculate_species_H_S(T, "NO2")
    #     H_O2, S_O2 = self.calculate_species_H_S(T, "O2")

    #     delta_H = H_NO2 - H_NO - 0.5 * H_O2
    #     delta_S = S_NO2 - S_NO - 0.5 * S_O2

    #     delta_G = delta_H * 1000 - T * delta_S

    #     log_Kp = -delta_G / (self.R * T)

    #     return torch.exp(torch.clamp(log_Kp, min=-80, max=80))

    # def calculate_no2_equilibrium_selectivity(self, T: torch.Tensor, o2_fraction: torch.Tensor, pressure_bar: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    #     Kp = self.calculate_no2_equilibrium_constant(T)
    #     p_O2 = o2_fraction * pressure_bar
    #     ratio = Kp * torch.sqrt(torch.clamp(p_O2, min=eps))
    #     return ratio / (1 + ratio)

    def calculate_no2_equilibrium_constant(self, T: torch.Tensor) -> torch.Tensor:
        delta_G = self.DELTA_H_NO2_DECOMP - T * self.DELTA_S_NO2_DECOMP
        log_Kp = -delta_G / (self.R * T)
        return torch.exp(torch.clamp(log_Kp, min=-80, max=80))

    def calculate_no2_equilibrium_selectivity(self, T: torch.Tensor, o2_fraction: torch.Tensor, pressure_bar: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
        Kp = self.calculate_no2_equilibrium_constant(T)
        p_O2 = o2_fraction * pressure_bar
        o2_activity = torch.clamp(p_O2 / self.P_STANDARD_BAR, min=eps)
        no_to_no2_ratio = torch.sqrt(Kp / o2_activity)
        return 1.0 / (1.0 + no_to_no2_ratio)


    def calculate_no2_equilibrium_flow(self, T: torch.Tensor, o2_fraction: torch.Tensor, nox_flow: torch.Tensor, pressure_bar: float = 1.0) -> torch.Tensor:
        S_eq = self.calculate_no2_equilibrium_selectivity(T=T, o2_fraction=o2_fraction, pressure_bar=pressure_bar)
        return S_eq * nox_flow

    def calculate_no2_selectivity(self, no2_flow: torch.Tensor, nox_flow: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return no2_flow / (nox_flow + eps)