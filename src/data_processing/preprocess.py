import numpy as np
import pandas as pd
from ..physics_calculator import PhysicsCalculator
from typing import List, Optional

class Preprocessor:
    METAL_PREFIX = "sup_"

    NON_METAL_SUP_COLS = {
        "sup_calcination_temp",
        "sup_calcination_time",
        "sup_crystallite_size_nm",
        "sup_M_to_O_ratio",
    }


    def __init__(self, initial_nox_mode="measured_species"):
        self.physics_calculator = PhysicsCalculator()
        self.initial_nox_mode = initial_nox_mode

    def preprocess(
        self,
        experiments: pd.DataFrame,
        materials: pd.DataFrame,
        allowed_elements: Optional[List[str]] = None
    ):
        experiments = experiments.copy()
        materials = materials.copy()

        materials = self.preprocess_materials(materials, allowed_elements)
        experiments = self.preprocess_experiments(experiments)

        return experiments, materials


    def preprocess_materials(
        self,
        df: pd.DataFrame,
        allowed_elements=None,
    ):
        df = df.copy()

        metal_cols = [
            col for col in df.columns
            if col.startswith(self.METAL_PREFIX)
            and col not in self.NON_METAL_SUP_COLS
        ]

        # Convert element columns to numeric and treat missing elements as absent
        df[metal_cols] = (
            df[metal_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

        # Remove materials containing elements outside the allowed set
        if allowed_elements is not None:
            allowed_elements = set(allowed_elements)

            disallowed_cols = [
                col for col in metal_cols
                if col.removeprefix(self.METAL_PREFIX)
                not in allowed_elements
            ]

            if disallowed_cols:
                has_disallowed_element = (
                    df[disallowed_cols] != 0
                ).any(axis=1)

                df = df.loc[~has_disallowed_element].copy()

        df = self.convert_support_composition_to_fraction(df)

        return df


    def preprocess_experiments(self, df: pd.DataFrame):
        df = self.convert_experiment_units(df)
        df = self.add_inlet_flows(df)
        df = self.add_initial_conditions(df)

        return df
    
    def convert_support_composition_to_fraction(self, df: pd.DataFrame):
        metal_cols = [
            col for col in df.columns
            if col.startswith(self.METAL_PREFIX)
            and col not in self.NON_METAL_SUP_COLS
        ]

        total = df[metal_cols].sum(axis=1)
        valid = total > 0
        df.loc[valid, metal_cols] = df.loc[valid, metal_cols].div(total.loc[valid], axis=0)
        return df
    
    def convert_experiment_units(self, df: pd.DataFrame):
        # Fractions
        df["O2_fraction"] = df["O2_vol%"] / 100

        df["soot_oxidation_conversion"] /= 100
        df["no2_fraction_of_nox"] = df["no2_%_of_nox"] / 100
        df["total_nox_conversion"] = df["total_nox_conversion_%"] / 100
        df["soot_oxidation_co2_selectivity"] /= 100

        # Temperature
        df["temperature_K"] = df["temperature"] + self.physics_calculator.K
        df["start_temp_K"] = df["start_temp"] + self.physics_calculator.K

        # Keep ramp rate in K/min because delta K == delta C
        df["ramp_rate_K_min"] = df["ramp_rate_C_min"]

        df["mass_soot_mg"] = df["mass_soot"] * 1000

        return df
    

    def add_inlet_flows(self, df: pd.DataFrame):
        df["F_total_mol_min"] = df["gas_flow_ml_min"] / self.physics_calculator.MOLAR_VOLUME_STP_ML
        df["F_NO_in"] = df["NO_ppm"] * 1e-6 * df["F_total_mol_min"]
        df["F_NO2_in"] = df["NO2_ppm"] * 1e-6 * df["F_total_mol_min"]
        df["F_NOx_in"] = df["F_NO_in"].fillna(0) + df["F_NO2_in"].fillna(0)
        return df
    
    def add_initial_conditions(self, df):
        df = df.copy()

        first = (
            df.sort_values(["_id", "temperature"])
            .drop_duplicates(subset="_id", keep="first")
            .copy()
        )

        if self.initial_nox_mode == "measured_species":
            initial = self._initial_nox_from_measured_species(first)

        elif self.initial_nox_mode == "first_selectivity":
            initial = self._initial_nox_from_first_selectivity(df, first)

        else:
            raise ValueError(
                f"Unknown initial_nox_mode: {self.initial_nox_mode}. "
                "Expected 'measured_species' or 'first_selectivity'."
            )

        df = df.merge(
            initial,
            on="_id",
            how="left",
            validate="many_to_one",
        )

        df["F_NO_initial"] = (
            df["NO_initial_ppm"]
            * 1e-6
            * df["F_total_mol_min"]
        )

        df["F_NO2_initial"] = (
            df["NO2_initial_ppm"]
            * 1e-6
            * df["F_total_mol_min"]
        )

        df["F_NOx_initial"] = (
            df["F_NO_initial"]
            + df["F_NO2_initial"]
        )

        return df
    
    def merge_materials_experiments(
        self,
        experiments: pd.DataFrame,
        materials: pd.DataFrame,
        how="inner"
    ) -> pd.DataFrame:
        return experiments.merge(
            materials,
            left_on="material_id",
            right_on="_id",
            how=how,
            suffixes=("_experiment", "_material"),
        )
    
    def _initial_nox_from_measured_species(self, first):
        inlet_no = first["NO_ppm"].fillna(0.0)
        inlet_no2 = first["NO2_ppm"].fillna(0.0)
        inlet_nox = inlet_no + inlet_no2

        initial_no = inlet_no.copy()
        initial_no2 = inlet_no2.copy()
        initial_nox = inlet_nox.copy()

        has_initial_no2 = first["no2_ppm"].notna()

        has_initial_no_and_no2 = (
            first["no_ppm"].notna()
            & first["no2_ppm"].notna()
        )

        initial_no.loc[has_initial_no_and_no2] = (
            first.loc[has_initial_no_and_no2, "no_ppm"]
        )

        initial_no2.loc[has_initial_no2] = (
            first.loc[has_initial_no2, "no2_ppm"]
        )

        has_initial_nox = first["nox_ppm"].notna()

        initial_nox.loc[has_initial_nox & has_initial_no2] = (
            first.loc[has_initial_nox & has_initial_no2, "nox_ppm"]
        )

        calculate_nox = has_initial_no2 & ~has_initial_nox

        initial_nox.loc[calculate_nox] = (
            initial_no.loc[calculate_nox]
            + initial_no2.loc[calculate_nox]
        )

        infer_no = (
            has_initial_no2
            & has_initial_nox
            & first["no_ppm"].isna()
        )

        initial_no.loc[infer_no] = (
            initial_nox.loc[infer_no]
            - initial_no2.loc[infer_no]
        )

        source = np.where(
            has_initial_no2,
            "initial_measurement",
            "reported_inlet",
        )

        result = first[["_id"]].copy()

        result["NO_initial_ppm"] = initial_no
        result["NO2_initial_ppm"] = initial_no2
        result["NOx_initial_ppm"] = initial_nox
        result["initial_NOx_source"] = source

        result["S_NO2_initial"] = np.where(
            initial_nox > 0,
            initial_no2 / initial_nox,
            np.nan,
        )

        return result
    
    def _initial_nox_from_first_selectivity(self, df, first):
        inlet_no = pd.to_numeric(first["NO_ppm"], errors="coerce").fillna(0.0)
        inlet_no2 = pd.to_numeric(first["NO2_ppm"], errors="coerce").fillna(0.0)

        reported_nox = inlet_no + inlet_no2

        # If an explicit inlet total NOx exists, prefer it.
        if "NOx_ppm" in first.columns:
            explicit_nox = pd.to_numeric(first["NOx_ppm"], errors="coerce")
            reported_nox = explicit_nox.where(explicit_nox.notna(), reported_nox)

        # Find the first measured NO2 selectivity for each experiment.
        first_selectivity = (
            df.loc[df["no2_fraction_of_nox"].notna(), ["_id", "temperature", "no2_fraction_of_nox"]]
            .sort_values(["_id", "temperature"])
            .drop_duplicates("_id", keep="first")
            .set_index("_id")["no2_fraction_of_nox"]
            .reindex(first["_id"])
        )

        first_selectivity.index = first.index

        valid_selectivity = (
            first_selectivity.notna()
            & first_selectivity.between(0.0, 1.0)
            & (reported_nox > 0)
        )

        # Default/fallback is the nominal reported inlet composition.
        initial_no = inlet_no.copy()
        initial_no2 = inlet_no2.copy()
        initial_nox = reported_nox.copy()

        # For this model, preserve reported total NOx but initialise its NO/NO2
        # split using the first experimentally observed selectivity.
        initial_no2.loc[valid_selectivity] = (
            reported_nox.loc[valid_selectivity]
            * first_selectivity.loc[valid_selectivity]
        )

        initial_no.loc[valid_selectivity] = (
            reported_nox.loc[valid_selectivity]
            * (1.0 - first_selectivity.loc[valid_selectivity])
        )

        source = np.where(
            valid_selectivity,
            "first_selectivity+reported_inlet_total",
            "reported_inlet",
        )

        result = first[["_id"]].copy()

        result["NO_initial_ppm"] = initial_no
        result["NO2_initial_ppm"] = initial_no2
        result["NOx_initial_ppm"] = initial_nox
        result["initial_NOx_source"] = source

        result["S_NO2_initial"] = np.where(
            initial_nox > 0,
            initial_no2 / initial_nox,
            np.nan,
        )

        return result