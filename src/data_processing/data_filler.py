from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FillReport:
    """Summary of values created by DataFiller."""
    counts: Dict[str, int] = field(default_factory=dict)

    def add(self, field_name: str, n: int) -> None:
        if n:
            self.counts[field_name] = self.counts.get(field_name, 0) + int(n)

    @property
    def total_filled(self) -> int:
        return sum(self.counts.values())

    def print(self) -> None:
        print(f"Filled {self.total_filled} values")
        if not self.counts:
            print("No values could be derived.")
            return

        for name, count in sorted(
            self.counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(f"  {name}: {count}")


class DataFiller:
    """
    Deterministically fill chemically related fields in a row-per-temperature
    soot/NOx experiment DataFrame.

    Important principles
    --------------------
    1. Existing non-null values are NEVER overwritten.
    2. No interpolation or model-based imputation is performed.
    3. Relationships are applied iteratively because one derived field may
       unlock another deterministic calculation.
    4. Calculations are performed row-by-row, so values never leak between
       experiments or temperatures.

    Percentage columns in this dataset are assumed to use percentage points,
    e.g. 85 means 85%, not 0.85.
    """

    IN_NO = "NO_ppm"
    IN_NO2 = "NO2_ppm"
    IN_NOX = "NOx_ppm"
    OUT_NO = "no_ppm"
    OUT_NO2 = "no2_ppm"
    OUT_NOX = "nox_ppm"
    NO2_SELECTIVITY = "no2_%_of_nox"
    NOX_CONVERSION = "total_nox_conversion_%"
    CO2_PPM = "soot_oxidation_co2_concentration_ppm"
    CO_PPM = "soot_oxidation_co_concentration_ppm"
    COX_PPM = "soot_oxidation_cox_concentration_ppm"
    CO2_PERCENT = "soot_oxidation_co2_concentration_%"
    COX_PERCENT = "soot_oxidation_cox_concentration_%"
    CO2_SELECTIVITY = "soot_oxidation_co2_selectivity"
    SOOT_CONVERSION = "soot_oxidation_conversion"
    MASS_SOOT = "mass_soot"
    MASS_SOOT_REMAINING = "mass_soot_remaining_mg"
    MASS_SOOT_CONVERTED = "mass_soot_converted_mg"
    HAS_SOOT = "has_soot"
    HAS_NOX = "has_NOx"
    TIGHT_CONTACT = "tight_contact"

    def __init__(
        self,
        *,
        add_provenance: bool = True,
        max_iterations: int = 10,
        clip_small_negative: float = 1e-12,
    ):
        self.add_provenance = add_provenance
        self.max_iterations = max_iterations
        self.clip_small_negative = clip_small_negative
        self.report = FillReport()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fill(self, df: pd.DataFrame, *, inplace: bool = False) -> pd.DataFrame:
        """
        Fill all deterministic values that can be inferred.

        Parameters
        ----------
        df
            Row-per-temperature experimental DataFrame.
        inplace
            If False (default), return a copy. If True, modify df directly.

        Returns
        -------
        pandas.DataFrame
            DataFrame with deterministic gaps filled.
        """
        if not inplace:
            df = df.copy()

        self.report = FillReport()

        self._ensure_derived_columns(df)

        # Structural zeros from known experiment composition. These are applied
        # before the algebraic fills so they can unlock additional deterministic
        # relationships such as COx = CO2 + CO.
        self._fill_structural_zeros(df)

        # Several equations are invertible. Iterate until an entire pass
        # produces no new values.
        for _ in range(self.max_iterations):
            before = int(df.notna().sum().sum())

            self._fill_inlet_nox(df)
            self._fill_co2_unit_conversions(df)
            self._fill_cox_unit_conversions(df)
            self._fill_carbon_balance(df)
            self._fill_nox_balance(df)
            self._fill_soot_mass(df)

            after = int(df.notna().sum().sum())
            if after == before:
                break

        return df

    @staticmethod
    def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return pd.to_numeric(df[column], errors="coerce")

    @staticmethod
    def _missing(df: pd.DataFrame, column: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series(True, index=df.index)
        return df[column].isna()

    def _ensure_column(self, df: pd.DataFrame, column: str) -> None:
        if column not in df.columns:
            df[column] = np.nan

        if self.add_provenance:
            source_col = f"{column}__calculated_from"
            if source_col not in df.columns:
                df[source_col] = pd.Series(pd.NA, index=df.index, dtype="object")

    def _ensure_derived_columns(self, df: pd.DataFrame) -> None:
        # Columns that either may not be present in the raw DataFrame or are
        # explicitly created by the filler.
        for column in (
            self.IN_NOX,
            self.CO2_PPM,
            self.CO_PPM,
            self.COX_PPM,
            self.CO2_PERCENT,
            self.COX_PERCENT,
            self.CO2_SELECTIVITY,
            self.OUT_NOX,
            self.OUT_NO,
            self.OUT_NO2,
            self.NO2_SELECTIVITY,
            self.NOX_CONVERSION,
            self.MASS_SOOT_REMAINING,
            self.MASS_SOOT_CONVERTED,
        ):
            self._ensure_column(df, column)

    def _assign(
        self,
        df: pd.DataFrame,
        column: str,
        mask: pd.Series,
        values: pd.Series,
        source: str,
    ) -> int:
        """
        Fill only genuinely missing target values and finite calculated values.
        """
        self._ensure_column(df, column)

        values = pd.to_numeric(values, errors="coerce")
        finite = pd.Series(
            np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)),
            index=df.index,
        )

        fill_mask = mask & self._missing(df, column) & finite
        n = int(fill_mask.sum())

        if n == 0:
            return 0

        cleaned = values.copy()

        # Remove tiny floating point negatives such as -1e-15.
        tiny_negative = (
            (cleaned < 0)
            & (cleaned >= -self.clip_small_negative)
        )
        cleaned.loc[tiny_negative] = 0.0

        df.loc[fill_mask, column] = cleaned.loc[fill_mask]

        if self.add_provenance:
            df.loc[
                fill_mask,
                f"{column}__calculated_from",
            ] = source

        self.report.add(column, n)
        return n

    @staticmethod
    def _false_mask(df: pd.DataFrame, column: str) -> pd.Series:
        """
        Rows explicitly marked False/0.

        Missing flags are deliberately NOT treated as False, because absence of
        metadata should never create artificial zeros.
        """
        if column not in df.columns:
            return pd.Series(False, index=df.index)

        values = df[column]
        if pd.api.types.is_bool_dtype(values):
            return values.eq(False) & values.notna()

        numeric = pd.to_numeric(values, errors="coerce")
        numeric_false = numeric.eq(0) & numeric.notna()

        strings = values.astype("string").str.strip().str.lower()
        string_false = strings.isin({"false", "f", "no", "n"})

        return numeric_false | string_false


    def _fill_structural_zeros(self, df: pd.DataFrame) -> None:
        """
        Fill quantities that are structurally zero because a reactant/component
        is explicitly absent.

        If has_soot == False:
          - mass_soot_remaining = 0
          - mass_soot_converted = 0
          - soot-derived CO2 concentration = 0
          - soot-derived CO concentration = 0
          - consequently COx concentration can become 0 through the normal
            carbon-balance rules

        If has_NOx == False:
          - inlet NOx, NO and NO2 concentrations = 0
          - outlet NOx, NO and NO2 concentrations = 0

        NO2 selectivity and total NOx conversion are NOT set to zero, because
        those ratios are undefined when no NOx exists.
        """
        no_soot = self._false_mask(df, self.HAS_SOOT)
        no_nox = self._false_mask(df, self.HAS_NOX)

        df["mass_silica_beads"] = df["mass_silica_beads"].fillna(0)
        # No soot -> no soot mass present/converted and no soot oxidation carbon
        # products. Fill both ppm and % CO2 representations so downstream code
        # remains consistent regardless of which representation is used.
        for column, source in (
            (
                self.MASS_SOOT,
                "has_soot == False",
            ),
            (
                self.MASS_SOOT_REMAINING,
                "has_soot == False",
            ),
            (
                self.MASS_SOOT_CONVERTED,
                "has_soot == False",
            ),
            (
                self.CO2_PPM,
                "has_soot == False",
            ),
            (
                self.CO2_PERCENT,
                "has_soot == False",
            ),
            (
                self.CO_PPM,
                "has_soot == False",
            ),
            (
                self.TIGHT_CONTACT,
                "has_soot == False",
            ),
        ):
            self._assign(
                df,
                column,
                no_soot,
                pd.Series(0.0, index=df.index),
                source,
            )

        # No inlet NOx -> all absolute NOx species concentrations are zero.
        # Ratio quantities remain NaN/undefined.
        for column in (
            self.IN_NOX,
            self.IN_NO,
            self.IN_NO2,
            self.OUT_NOX,
            self.OUT_NO,
            self.OUT_NO2,
        ):
            self._assign(
                df,
                column,
                no_nox,
                pd.Series(0.0, index=df.index),
                "has_NOx == False",
            )

    def _fill_inlet_nox(self, df: pd.DataFrame) -> None:
        """
        NOx_in = NO_in + NO2_in.

        In the source data, a null inlet species normally means that species
        was not fed. Therefore, if at least one of NO_ppm / NO2_ppm is known,
        the missing companion is treated as zero for this SUM ONLY.

        If both are null, NOx_ppm remains null.
        """
        no = self._numeric(df, self.IN_NO)
        no2 = self._numeric(df, self.IN_NO2)

        at_least_one_known = no.notna() | no2.notna()
        values = no.fillna(0.0) + no2.fillna(0.0)

        self._assign(
            df,
            self.IN_NOX,
            at_least_one_known,
            values,
            "NO_ppm + NO2_ppm (missing inlet species treated as 0)",
        )

    def _fill_co2_unit_conversions(self, df: pd.DataFrame) -> None:
        """
        Exact gas concentration conversion:
            1 vol% = 10,000 ppm
        """
        co2_ppm = self._numeric(df, self.CO2_PPM)
        co2_pct = self._numeric(df, self.CO2_PERCENT)

        self._assign(
            df,
            self.CO2_PPM,
            co2_pct.notna(),
            co2_pct * 10_000.0,
            f"{self.CO2_PERCENT} * 10000",
        )
        co2_ppm = self._numeric(df, self.CO2_PPM)

        self._assign(
            df,
            self.CO2_PERCENT,
            co2_ppm.notna(),
            co2_ppm / 10_000.0,
            f"{self.CO2_PPM} / 10000",
        )

    def _fill_cox_unit_conversions(self, df: pd.DataFrame) -> None:
        cox_ppm = self._numeric(df, self.COX_PPM)
        cox_pct = self._numeric(df, self.COX_PERCENT)

        self._assign(
            df,
            self.COX_PPM,
            cox_pct.notna(),
            cox_pct * 10_000.0,
            f"{self.COX_PERCENT} * 10000",
        )

        cox_ppm = self._numeric(df, self.COX_PPM)

        self._assign(
            df,
            self.COX_PERCENT,
            cox_ppm.notna(),
            cox_ppm / 10_000.0,
            f"{self.COX_PPM} / 10000",
        )

    def _fill_carbon_balance(self, df: pd.DataFrame) -> None:
        """
        Definitions:
            COx = CO2 + CO
            S_CO2 = 100 * CO2 / COx

        These equations are used in every algebraically invertible direction.
        """
        co2 = self._numeric(df, self.CO2_PPM)
        co = self._numeric(df, self.CO_PPM)
        cox = self._numeric(df, self.COX_PPM)
        sel = self._numeric(df, self.CO2_SELECTIVITY)

        # COx from measured CO2 + CO.
        self._assign(
            df,
            self.COX_PPM,
            co2.notna() & co.notna(),
            co2 + co,
            f"{self.CO2_PPM} + {self.CO_PPM}",
        )

        # Refresh.
        cox = self._numeric(df, self.COX_PPM)

        # CO2 from COx and selectivity.
        valid_sel = sel.notna() & sel.between(0, 100, inclusive="both")
        self._assign(
            df,
            self.CO2_PPM,
            cox.notna() & valid_sel,
            cox * sel / 100.0,
            f"{self.COX_PPM} * {self.CO2_SELECTIVITY} / 100",
        )

        # CO from COx and selectivity.
        self._assign(
            df,
            self.CO_PPM,
            cox.notna() & valid_sel,
            cox * (1.0 - sel / 100.0),
            f"{self.COX_PPM} * (1 - {self.CO2_SELECTIVITY}/100)",
        )

        # Refresh after possible CO2 / CO fill.
        co2 = self._numeric(df, self.CO2_PPM)
        co = self._numeric(df, self.CO_PPM)
        valid_nonzero_sel = sel.notna() & (sel > 0) & (sel <= 100)
        self._assign(
            df,
            self.CO_PPM,
            co2.notna() & valid_nonzero_sel,
            co2 * (100.0 / sel - 1.0),
            f"{self.CO2_PPM} and {self.CO2_SELECTIVITY}",
        )

        # Extra inverse: infer CO2 from CO + selectivity.
        # CO2 = CO * S / (100-S)
        valid_not_100 = sel.notna() & (sel >= 0) & (sel < 100)
        denominator = 100.0 - sel
        self._assign(
            df,
            self.CO2_PPM,
            co.notna() & valid_not_100,
            co * sel / denominator,
            f"{self.CO_PPM} and {self.CO2_SELECTIVITY}",
        )

        # Refresh all carbon quantities.
        co2 = self._numeric(df, self.CO2_PPM)
        co = self._numeric(df, self.CO_PPM)
        cox = self._numeric(df, self.COX_PPM)

        # CO2 selectivity from CO2 + CO.
        total = co2 + co
        self._assign(
            df,
            self.CO2_SELECTIVITY,
            co2.notna() & co.notna() & (total > 0),
            100.0 * co2 / total,
            f"100 * {self.CO2_PPM} / ({self.CO2_PPM} + {self.CO_PPM})",
        )

        # Or directly from CO2 / COx.
        self._assign(
            df,
            self.CO2_SELECTIVITY,
            co2.notna() & cox.notna() & (cox > 0),
            100.0 * co2 / cox,
            f"100 * {self.CO2_PPM} / {self.COX_PPM}",
        )

        # Percentage-side deterministic relationship.
        co2_pct = self._numeric(df, self.CO2_PERCENT)
        cox_pct = self._numeric(df, self.COX_PERCENT)
        sel = self._numeric(df, self.CO2_SELECTIVITY)

        # CO2% from COx% and selectivity.
        valid_sel = sel.notna() & sel.between(0, 100, inclusive="both")
        self._assign(
            df,
            self.CO2_PERCENT,
            cox_pct.notna() & valid_sel,
            cox_pct * sel / 100.0,
            f"{self.COX_PERCENT} * {self.CO2_SELECTIVITY} / 100",
        )

        # Selectivity from CO2% / COx%.
        self._assign(
            df,
            self.CO2_SELECTIVITY,
            co2_pct.notna() & cox_pct.notna() & (cox_pct > 0),
            100.0 * co2_pct / cox_pct,
            f"100 * {self.CO2_PERCENT} / {self.COX_PERCENT}",
        )

    def _fill_nox_balance(self, df: pd.DataFrame) -> None:
        """
        Definitions:
            NOx_out = NO_out + NO2_out
            X_NOx = 100 * (1 - NOx_out / NOx_in)
            S_NO2 = 100 * NO2_out / NOx_out

        Every algebraically invertible, physically well-defined direction is
        used to fill missing values.
        """
        nox_in = self._numeric(df, self.IN_NOX)
        no = self._numeric(df, self.OUT_NO)
        no2 = self._numeric(df, self.OUT_NO2)
        nox = self._numeric(df, self.OUT_NOX)
        conv = self._numeric(df, self.NOX_CONVERSION)
        sel = self._numeric(df, self.NO2_SELECTIVITY)

        # Outlet NOx from NO + NO2.
        self._assign(
            df,
            self.OUT_NOX,
            no.notna() & no2.notna(),
            no + no2,
            f"{self.OUT_NO} + {self.OUT_NO2}",
        )

        nox = self._numeric(df, self.OUT_NOX)
        # NOx_out = NOx_in * (1 - conversion/100)
        self._assign(
            df,
            self.OUT_NOX,
            nox_in.notna() & conv.notna(),
            nox_in * (1.0 - conv / 100.0),
            f"{self.IN_NOX} * (1 - {self.NOX_CONVERSION}/100)",
        )

        nox = self._numeric(df, self.OUT_NOX)
        valid_sel = sel.notna() & sel.between(0, 100, inclusive="both")

        self._assign(
            df,
            self.OUT_NO2,
            nox.notna() & valid_sel,
            nox * sel / 100.0,
            f"{self.OUT_NOX} * {self.NO2_SELECTIVITY}/100",
        )

        self._assign(
            df,
            self.OUT_NO,
            nox.notna() & valid_sel,
            nox * (1.0 - sel / 100.0),
            f"{self.OUT_NOX} * (1 - {self.NO2_SELECTIVITY}/100)",
        )

        no = self._numeric(df, self.OUT_NO)
        no2 = self._numeric(df, self.OUT_NO2)
        nox = self._numeric(df, self.OUT_NOX)

        # Extra: one outlet species from total minus the other.
        self._assign(
            df,
            self.OUT_NO,
            nox.notna() & no2.notna(),
            nox - no2,
            f"{self.OUT_NOX} - {self.OUT_NO2}",
        )

        self._assign(
            df,
            self.OUT_NO2,
            nox.notna() & no.notna(),
            nox - no,
            f"{self.OUT_NOX} - {self.OUT_NO}",
        )

        no = self._numeric(df, self.OUT_NO)
        no2 = self._numeric(df, self.OUT_NO2)
        nox = self._numeric(df, self.OUT_NOX)

        self._assign(
            df,
            self.NO2_SELECTIVITY,
            no2.notna() & nox.notna() & (nox > 0),
            100.0 * no2 / nox,
            f"100 * {self.OUT_NO2} / {self.OUT_NOX}",
        )

        self._assign(
            df,
            self.NO2_SELECTIVITY,
            no.notna() & no2.notna() & ((no + no2) > 0),
            100.0 * no2 / (no + no2),
            f"100 * {self.OUT_NO2} / ({self.OUT_NO} + {self.OUT_NO2})",
        )

        # Extra: total NOx conversion from inlet and outlet total.
        self._assign(
            df,
            self.NOX_CONVERSION,
            nox_in.notna() & (nox_in != 0) & nox.notna(),
            100.0 * (1.0 - nox / nox_in),
            f"100 * (1 - {self.OUT_NOX}/{self.IN_NOX})",
        )

    def _fill_soot_mass(self, df: pd.DataFrame) -> None:
        mass_g = self._numeric(df, self.MASS_SOOT)
        mass_mg = mass_g * 1000.0

        conv = self._numeric(df, self.SOOT_CONVERSION)
        remaining = self._numeric(df, self.MASS_SOOT_REMAINING)
        converted = self._numeric(df, self.MASS_SOOT_CONVERTED)

        valid_mass = mass_mg.notna() & (mass_mg > 0)

        # m_converted from initial mass + conversion
        self._assign(
            df,
            self.MASS_SOOT_CONVERTED,
            valid_mass & conv.notna(),
            mass_mg * conv / 100.0,
            f"{self.MASS_SOOT} * 1000 * {self.SOOT_CONVERSION}/100",
        )

        # m_remaining from initial mass + conversion
        self._assign(
            df,
            self.MASS_SOOT_REMAINING,
            valid_mass & conv.notna(),
            mass_mg * (1.0 - conv / 100.0),
            f"{self.MASS_SOOT} * 1000 * (1 - {self.SOOT_CONVERSION}/100)",
        )

        remaining = self._numeric(df, self.MASS_SOOT_REMAINING)
        converted = self._numeric(df, self.MASS_SOOT_CONVERTED)

        # Conversion from remaining mass
        self._assign(
            df,
            self.SOOT_CONVERSION,
            valid_mass & remaining.notna(),
            100.0 * (1.0 - remaining / mass_mg),
            f"100 * (1 - {self.MASS_SOOT_REMAINING}/({self.MASS_SOOT}*1000))",
        )

        # Conversion from converted mass
        self._assign(
            df,
            self.SOOT_CONVERSION,
            valid_mass & converted.notna(),
            100.0 * converted / mass_mg,
            f"100 * {self.MASS_SOOT_CONVERTED}/({self.MASS_SOOT}*1000)",
        )

        # Remaining from initial - converted
        self._assign(
            df,
            self.MASS_SOOT_REMAINING,
            valid_mass & converted.notna(),
            mass_mg - converted,
            f"{self.MASS_SOOT}*1000 - {self.MASS_SOOT_CONVERTED}",
        )

        # Converted from initial - remaining
        self._assign(
            df,
            self.MASS_SOOT_CONVERTED,
            valid_mass & remaining.notna(),
            mass_mg - remaining,
            f"{self.MASS_SOOT}*1000 - {self.MASS_SOOT_REMAINING}",
        )



def fill_missing_experiment_data(
    df: pd.DataFrame,
    *,
    inplace: bool = False,
    add_provenance: bool = True,
) -> Tuple[pd.DataFrame, FillReport]:
    filler = DataFiller(add_provenance=add_provenance)
    filled = filler.fill(df, inplace=inplace)
    return filled, filler.report