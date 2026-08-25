CFG = {
    "data": {
        "fill_values": {
            "sup_calcination_temp": "mean",
            "sup_calcination_time": "median",
            "start_temp_K": 298
        }
    },
    "nn": {
        "hidden_dim": [16, 16],
        "output_dim": 6,
        "static_inputs": ["NO_initial_ppm", "NO2_initial_ppm", "O2_vol%", "mass_catalyst", "tight_contact", "mass_silica_beads", "gas_flow_ml_min", "sup_calcination_temp", "sup_calcination_time", "Sbet"], # elements automatically included
        "state": ["mass_soot_remaining_mg", "temperature_K"]
    },
    "physics": {
        "required_unscaled_inputs": ["mass_soot_remaining_mg", "ramp_rate_K_min", "temperature_K", "O2_fraction"],
        "targets": ["no2_fraction_of_nox", "nox_ppm", "no2_ppm", "no_ppm", "n2_ppm", "mass_soot_remaining_mg", "soot_oxidation_co2_concentration_ppm", "soot_oxidation_co_concentration_ppm", "soot_oxidation_co2_selectivity"]
    }
}