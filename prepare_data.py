from src.db.filters import material_filter, experiment_filter
from src.db import DB
from src.data_processing import DataFiller, Preprocessor, Scaler, DataSplitter
from pathlib import Path
import json

OUTDIR = "data/soot_conversion_from_cox_and_NO2_initial_from_S"
ALLOWED_ELEMENTS = None # None for all elements
SPLIT_OPTIONS = {
    "train_fraction": 0.7,
    "eval_fraction": 0.15,
    "test_fraction": 0.15,
    "split_on": "experiment",
    "experiment_id_col": "_id_experiment"
}
SEEDS = [1, 2, 3]

# CFG = {
#     "notes": "Best for modelling total NOx and CO/CO2 ppm",
#     "preprocessor": {"initial_nox_mode": "measured_species"},
#     "filler": {
#         "add_provenance": True,
#         "max_iterations": 10,
#         "clip_small_negative": 1e-12,
#         "derive_soot_conversion_from_cox": False
#     }
# }

CFG = {
    "notes": "Calculates initial NO2 from initial NOx * Selectivity (so dont use when considering total NOx), and uses COx ppm to fill in missing soot conversion using integrals (so dont use if fitting CO/CO2 ppm or mass balance)",
    "preprocessor": {"initial_nox_mode": "first_selectivity"},
    "filler": {
        "add_provenance": True,
        "max_iterations": 10,
        "clip_small_negative": 1e-12,
        "derive_soot_conversion_from_cox": True,
        "cox_endpoint_fraction": 0.05,
        "min_cox_points": 5
    }
}


if __name__ == "__main__":
    db = DB()
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "preprocessing_config.json", "w") as f:
        json.dump(CFG, f, indent=4)
    mats, exps = db.generate_dataframes(material_filter, experiment_filter, save_to_dir="data/base_dataframes")
    preprocessor = Preprocessor(**CFG["preprocessor"])
    filler = DataFiller(**CFG["filler"])

    filled_exps = filler.fill(exps)
    exps, mats = preprocessor.preprocess(filled_exps, mats, ALLOWED_ELEMENTS)
    merged = preprocessor.merge_materials_experiments(exps, mats, how="inner")

    for seed in SEEDS:
        split_cfg = SPLIT_OPTIONS.copy()
        split_cfg["random_state"] = seed
        splitter = DataSplitter(**split_cfg)
        train, val, test = splitter.split(merged)
        seed_outdir = Path(outdir / str(seed))
        seed_outdir.mkdir(exist_ok=True)
        train.to_csv(seed_outdir / "train.csv", index=False)
        test.to_csv(seed_outdir / "test.csv", index=False)
        val.to_csv(seed_outdir / "validation.csv", index=False)