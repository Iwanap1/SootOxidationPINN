from src.db.filters import material_filter, experiment_filter
from src.db import DB
from src.data_processing import DataFiller, Preprocessor, Scaler, DataSplitter
from pathlib import Path

OUTDIR = "data/experiment_split"
ALLOWED_ELEMENTS = None # None for all elements
SPLIT_OPTIONS = {
    "train_fraction": 0.7,
    "eval_fraction": 0.15,
    "test_fraction": 0.15,
    "split_on": "experiment",
    "experiment_id_col": "_id_experiment"
}
SEEDS = [1, 2, 3]

if __name__ == "__main__":
    db = DB()
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)
    mats, exps = db.generate_dataframes(material_filter, experiment_filter, save_to_dir="data/base_dataframes")
    preprocessor = Preprocessor()
    filler = DataFiller()

    filled_exps = filler.fill(exps)
    exps, mats = preprocessor.preprocess(filled_exps, mats, ALLOWED_ELEMENTS)
    merged = preprocessor.merge_materials_experiments(exps, mats, how="inner")

    for seed in SEEDS:
        split_cfg = SPLIT_OPTIONS.copy()
        split_cfg["random_state"] = seed
        splitter = DataSplitter(**SPLIT_OPTIONS)
        train, val, test = splitter.split(merged)
        seed_outdir = Path(outdir / str(seed))
        seed_outdir.mkdir(exist_ok=True)
        train.to_csv(seed_outdir / "train.csv", index=False)
        test.to_csv(seed_outdir / "test.csv", index=False)
        val.to_csv(seed_outdir / "validation.csv", index=False)