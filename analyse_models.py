from src.data_processing import ExperimentDataset
from src.analyser import ModelAnalyser
from pathlib import Path
import torch
import json
import pickle
from sklearn.preprocessing import StandardScaler
from src.utils import load_model

model_dirs = [
    "models/only_s_nox_no_frac_carbon/experiment_split/1"
]

def analyse_model(model_dir: Path):
    test_dataset = torch.load(model_dir / "test_dataset.pt", weights_only=False)

    model = load_model(Path(model_dir), just_model=True)

    analyser = ModelAnalyser(model, test_dataset, outdir = Path(model_dir / "analysis"))
    results = analyser.analyse(n_experiments=20)

if __name__ == "__main__":
    for mdir in model_dirs:
        analyse_model(Path(mdir))