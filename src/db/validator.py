from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import re
from .fields import LIST_FIELDS, SOOT_LIST_FIELDS, NOX_LIST_FIELDS


class Validate:
    """Validate migration records before inserting them into the database.

    Validation is non-destructive: all problems are collected in ``errors`` and
    printed at the end.  Warnings are kept separately for suspicious, but not
    necessarily invalid, records.
    """
    # Only fields of the form imp_<element>_wt% count as an impregnated species.
    # This intentionally excludes imp_dispersion_CO/M, imp_calcination_temp, etc.
    IMP_SPECIES_RE = re.compile(r"^imp_[A-Z][a-z]?_wt%$")

    def __init__(self, data: List[Dict[str, Any]], *, print_report: bool = True):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.n_entries = len(data)
        self.n_experiments = 0

        for entry_index, entry in enumerate(data):
            self.validate(entry, entry_index)

        if print_report:
            self.print_report()

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def validate(self, entry: Dict[str, Any], entry_index: int) -> None:
        doi = entry.get("doi", "<missing DOI>")
        material = entry.get("material") or {}
        material_name = material.get("name", "<missing material name>")

        self._check_impregnation(material, doi, material_name)

        experiments = entry.get("experiments")
        if not isinstance(experiments, list):
            self._error(doi, material_name, None, "'experiments' is missing or is not a list")
            return

        for exp_index, exp in enumerate(experiments):
            self.n_experiments += 1
            if not isinstance(exp, dict):
                self._error(doi, material_name, exp_index, "experiment is not an object/dict")
                continue
            self.validate_experiment(exp, doi, material_name, exp_index)

    def validate_experiment(
        self,
        exp: Dict[str, Any],
        doi: str,
        material_name: str,
        exp_index: int,
    ) -> None:
        self._check_mandatory_fields(exp, doi, material_name, exp_index)
        self._same_length_lists(exp, doi, material_name, exp_index)
        self._check_soot(exp, doi, material_name, exp_index)
        self._check_nox(exp, doi, material_name, exp_index)
        self._check_mass_consistency(exp, doi, material_name, exp_index)

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_blank(value: Any) -> bool:
        """True for None, empty strings and NaN; False for numeric zero."""
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        if isinstance(value, float) and math.isnan(value):
            return True
        return False

    @classmethod
    def _has_value(cls, value: Any) -> bool:
        """Whether a scalar/list contains actual data."""
        if cls._is_blank(value):
            return False
        if isinstance(value, (list, tuple)):
            return len(value) > 0
        return True

    @classmethod
    def _positive_number(cls, value: Any) -> bool:
        if cls._is_blank(value) or isinstance(value, bool):
            return False
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _parse_ratio(value: Any) -> Optional[Tuple[float, float]]:
        """Parse soot:catalyst ratio such as '1:10'."""
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        try:
            left, right = str(value).strip().split(":")
            soot_part, catalyst_part = float(left), float(right)
            if soot_part <= 0 or catalyst_part <= 0:
                return None
            return soot_part, catalyst_part
        except (ValueError, TypeError):
            return None

    @classmethod
    def _list_has_values(cls, exp: Dict[str, Any], field: str) -> bool:
        value = exp.get(field)
        if cls._is_blank(value):
            return False
        if isinstance(value, list):
            return len(value) > 0
        # Some fields, notably soot_oxidation_co2_selectivity, can be scalar.
        return True

    def _context(self, doi: str, material_name: str, exp_index: Optional[int]) -> str:
        context = f"DOI={doi} | Material={material_name}"
        if exp_index is not None:
            context += f" | Experiment={exp_index + 1}"
        return context

    def _error(self, doi: str, material_name: str, exp_index: Optional[int], message: str) -> None:
        self.errors.append(f"{self._context(doi, material_name, exp_index)} | {message}")

    def _warning(self, doi: str, material_name: str, exp_index: Optional[int], message: str) -> None:
        self.warnings.append(f"{self._context(doi, material_name, exp_index)} | {message}")

    # ------------------------------------------------------------------
    # Validation rules
    # ------------------------------------------------------------------
    def _check_mandatory_fields(
        self,
        exp: Dict[str, Any],
        doi: str,
        material_name: str,
        exp_index: int,
    ) -> None:
        """Check fields that every experiment must define.

        Required for every experiment:
          - gas_flow_ml_min: numeric and > 0
          - O2_vol%: numeric and >= 0
          - ramp_rate_C_min: numeric and > 0

        Additionally, if soot is present, tight_contact must be explicitly
        provided as 0 or 1. Numeric zero is therefore valid and is not treated
        as a missing value.
        """

        # Flow rate
        flow = exp.get("gas_flow_ml_min")
        if self._is_blank(flow):
            self._error(
                doi,
                material_name,
                exp_index,
                "'gas_flow_ml_min' is mandatory and cannot be None/blank",
            )
        elif not self._positive_number(flow):
            self._error(
                doi,
                material_name,
                exp_index,
                f"'gas_flow_ml_min' must be a positive number (got {flow!r})",
            )

        # Oxygen concentration
        o2 = exp.get("O2_vol%")
        if self._is_blank(o2):
            self._error(
                doi,
                material_name,
                exp_index,
                "'O2_vol%' is mandatory and cannot be None/blank",
            )
        else:
            try:
                o2_value = float(o2)
                if not math.isfinite(o2_value) or o2_value < 0:
                    raise ValueError
            except (TypeError, ValueError):
                self._error(
                    doi,
                    material_name,
                    exp_index,
                    f"'O2_vol%' must be a non-negative number (got {o2!r})",
                )

        # Temperature ramp rate
        ramp = exp.get("ramp_rate_C_min")
        if self._is_blank(ramp):
            self._error(
                doi,
                material_name,
                exp_index,
                "'ramp_rate_C_min' is mandatory and cannot be None/blank",
            )
        elif not self._positive_number(ramp):
            self._error(
                doi,
                material_name,
                exp_index,
                f"'ramp_rate_C_min' must be a positive number (got {ramp!r})",
            )

        # Determine soot presence the same way as _check_soot().
        has_soot = (
            self._positive_number(exp.get("mass_soot"))
            or not self._is_blank(exp.get("soot_to_catalyst_mass_ratio"))
        )

        if has_soot:
            tight_contact = exp.get("tight_contact")

            if self._is_blank(tight_contact):
                self._error(
                    doi,
                    material_name,
                    exp_index,
                    "soot is present, so 'tight_contact' is mandatory and cannot be None/blank",
                )
            elif tight_contact not in (0, 1, False, True):
                self._error(
                    doi,
                    material_name,
                    exp_index,
                    f"when soot is present, 'tight_contact' must be 0 or 1 (got {tight_contact!r})",
                )

    def _same_length_lists(
        self,
        exp: Dict[str, Any],
        doi: str,
        material_name: str,
        exp_index: int,
    ) -> None:
        temps = exp.get("temps")
        if not isinstance(temps, list):
            self._error(doi, material_name, exp_index, "'temps' is missing or is not a list")
            return

        n_temps = len(temps)
        if n_temps == 0:
            self._warning(doi, material_name, exp_index, "no temperatures provided")

        for field in LIST_FIELDS:
            if field not in exp:
                continue

            value = exp[field]
            if self._is_blank(value):
                continue

            # CO2 selectivity is allowed to be a single summary value.
            if field == "soot_oxidation_co2_selectivity" and not isinstance(value, list):
                continue

            if not isinstance(value, list):
                self._error(doi, material_name, exp_index, f"'{field}' should be a list")
                continue

            if len(value) not in (0, n_temps):
                self._error(
                    doi,
                    material_name,
                    exp_index,
                    f"length mismatch for '{field}': temps has {n_temps}, field has {len(value)}",
                )

    def _check_soot(
        self,
        exp: Dict[str, Any],
        doi: str,
        material_name: str,
        exp_index: int,
    ) -> None:
        mass_soot = exp.get("mass_soot")
        ratio_raw = exp.get("soot_to_catalyst_mass_ratio")
        ratio = self._parse_ratio(ratio_raw)
        total_mass = exp.get("total_mass_g")
        mass_catalyst = exp.get("mass_catalyst")

        # Presence is determined from the experimental charge, not from result
        # arrays. This lets the validator catch soot data accidentally attached
        # to a no-soot experiment.
        has_soot = self._positive_number(mass_soot) or not self._is_blank(ratio_raw)

        if has_soot:
            # Rule 1: soot mass must be explicit OR resolvable.
            if not self._positive_number(mass_soot):
                if ratio is None:
                    self._error(
                        doi,
                        material_name,
                        exp_index,
                        f"soot is indicated but ratio '{ratio_raw}' is invalid and mass_soot is not provided",
                    )
                elif not (self._positive_number(total_mass) or self._positive_number(mass_catalyst)):
                    self._error(
                        doi,
                        material_name,
                        exp_index,
                        "soot is present, but mass_soot is missing and cannot be resolved: "
                        "need a valid soot:catalyst ratio plus total_mass_g or mass_catalyst",
                    )
        else:
            # Rules 2 and 5: no soot => no soot outputs or soot metadata.
            populated = [f for f in SOOT_LIST_FIELDS if self._list_has_values(exp, f)]
            if populated:
                self._error(
                    doi,
                    material_name,
                    exp_index,
                    "no soot is present, but soot-related data are populated: " + ", ".join(sorted(populated)),
                )

            for field in ("soot_name", "soot_supplier"):
                if not self._is_blank(exp.get(field)):
                    self._error(
                        doi,
                        material_name,
                        exp_index,
                        f"no soot is present, so '{field}' should be None/blank (got {exp.get(field)!r})",
                    )

            # These are also only meaningful for a soot-containing experiment.
            for field in ("soot_surface_area", "soot_oxidation_T_max", "tight_contact"):
                if not self._is_blank(exp.get(field)):
                    self._warning(
                        doi,
                        material_name,
                        exp_index,
                        f"no soot is present, but '{field}' is populated ({exp.get(field)!r})",
                    )

    def _check_nox(
        self,
        exp: Dict[str, Any],
        doi: str,
        material_name: str,
        exp_index: int,
    ) -> None:
        # NOx is considered fed if either inlet NO or inlet NO2 is > 0.
        has_nox = self._positive_number(exp.get("NO_ppm")) or self._positive_number(exp.get("NO2_ppm"))

        if not has_nox:
            populated = [f for f in NOX_LIST_FIELDS if self._list_has_values(exp, f)]
            if populated:
                self._error(
                    doi,
                    material_name,
                    exp_index,
                    "no NOx is present in the feed, but NOx-related data are populated: "
                    + ", ".join(sorted(populated)),
                )

            for field in ("no2_max_temp", "no2_max_%"):
                if not self._is_blank(exp.get(field)):
                    self._error(
                        doi,
                        material_name,
                        exp_index,
                        f"no NOx is present in the feed, so '{field}' should be None/blank",
                    )

    def _check_impregnation(self, material: Dict[str, Any], doi: str, material_name: str) -> None:
        imp_species_fields = [key for key in material if self.IMP_SPECIES_RE.match(key)]
        populated_species = [key for key in imp_species_fields if not self._is_blank(material.get(key))]

        if not populated_species:
            for field in ("imp_calcination_temp", "imp_calcination_time"):
                if not self._is_blank(material.get(field)):
                    self._error(
                        doi,
                        material_name,
                        None,
                        f"no impregnated species is present, so '{field}' should be None/blank "
                        f"(got {material.get(field)!r})",
                    )

    def _check_mass_consistency(
        self,
        exp: Dict[str, Any],
        doi: str,
        material_name: str,
        exp_index: int,
    ) -> None:
        """Extra consistency checks when enough mass information is supplied."""
        soot = exp.get("mass_soot")
        catalyst = exp.get("mass_catalyst")
        total = exp.get("total_mass_g")
        ratio_raw = exp.get("soot_to_catalyst_mass_ratio")
        ratio = self._parse_ratio(ratio_raw)

        # If soot + catalyst + total are all known, they should agree.
        if all(self._positive_number(v) for v in (soot, catalyst, total)):
            expected = float(soot) + float(catalyst)
            if not math.isclose(expected, float(total), rel_tol=1e-3, abs_tol=1e-6):
                self._warning(
                    doi,
                    material_name,
                    exp_index,
                    f"mass_soot + mass_catalyst = {expected:g} g, but total_mass_g = {float(total):g} g",
                )

        # If soot and catalyst masses and ratio are all supplied, check ratio.
        if ratio is not None and self._positive_number(soot) and self._positive_number(catalyst):
            soot_part, catalyst_part = ratio
            expected_ratio = soot_part / catalyst_part
            actual_ratio = float(soot) / float(catalyst)
            if not math.isclose(actual_ratio, expected_ratio, rel_tol=0.02, abs_tol=1e-9):
                self._warning(
                    doi,
                    material_name,
                    exp_index,
                    f"mass ratio is inconsistent: masses give soot:catalyst = {actual_ratio:g}, "
                    f"but soot_to_catalyst_mass_ratio is {ratio_raw!r}",
                )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def print_report(self) -> None:
        print(f"Validated {self.n_entries} entries / {self.n_experiments} experiments")
        print(f"Errors: {len(self.errors)} | Warnings: {len(self.warnings)}")

        if self.errors:
            print("\nERRORS")
            for i, error in enumerate(self.errors, 1):
                print(f"{i:>3}. {error}")

        if self.warnings:
            print("\nWARNINGS")
            for i, warning in enumerate(self.warnings, 1):
                print(f"{i:>3}. {warning}")


def validate_migration_payload(payload: Dict[str, Any], *, print_report: bool = True) -> Validate:
    """Convenience wrapper for JSON shaped like {'create': [...]}.

    Raises ValueError for the outer payload shape, while record-level validation
    errors are accumulated in the returned ``Validate`` object.
    """
    if not isinstance(payload, dict):
        raise ValueError("Migration payload must be a dictionary")
    if "create" not in payload or not isinstance(payload["create"], list):
        raise ValueError("Migration payload must contain a 'create' list")
    return Validate(payload["create"], print_report=print_report)