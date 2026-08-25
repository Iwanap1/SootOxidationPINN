from src.data_processing import ExperimentDataset
from src.architectures.total_nox_model import NOxSelectivityModel
from src.analyser import ModelAnalyser
from pathlib import Path
import torch
import json
import pickle
from sklearn.preprocessing import StandardScaler

model_dirs = [
    "models/full_balance/experiment_split/1",
    "models/full_balance/experiment_split/2"
]

def analyse_model(model_dir: Path):
    test_dataset = torch.load(model_dir / "test_dataset.pt", weights_only=False)

    with open(model_dir / "config.json", "r") as f:
        cfg = json.load(f)
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    model = NOxSelectivityModel(cfg, cfg["nn"]["input_dim"], scaler)
    state_dict = torch.load(model_dir / "best_model.pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    analyser = ModelAnalyser(model, test_dataset, outdir = Path(model_dir / "analysis"))
    results = analyser.analyse()

if __name__ == "__main__":
    for mdir in model_dirs:
        analyse_model(Path(mdir))