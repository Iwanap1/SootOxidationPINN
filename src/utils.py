import pandas as pd
from typing import Optional, Dict
from pathlib import Path
from .data_processing import Imputer, Scaler, ExperimentDataset
from .trainer import Trainer
import json, pickle
import math
import matplotlib.pyplot as plt
import torch
import sys
import importlib


ANOMALOUS_EXP_IDS = [
    "6a7d9fab7bffe6ebe23a4081", # fitted f_C was ~ 0.53 whilst others were in range 0.9 - 1.3
]

def load_dataframes(directory):
    train = pd.read_csv(directory / "train.csv")
    val = pd.read_csv(directory / "validation.csv")
    test = pd.read_csv(directory / "test.csv")
    val = val[~(val["_id_experiment"].isin(ANOMALOUS_EXP_IDS))]
    train = train[~(train["_id_experiment"].isin(ANOMALOUS_EXP_IDS))]
    test = test[~(test["_id_experiment"].isin(ANOMALOUS_EXP_IDS))]
    return train, val, test


def derive_input_cols(df, cfg):
    non_element_cols = {
        "sup_calcination_temp",
        "sup_calcination_time",
        "sup_crystallite_size_nm",
        "sup_M_to_O_ratio",
    }

    element_inputs = [
        col for col in df.columns
        if col.startswith("sup_")
        and not col.endswith("_scaled")
        and col not in non_element_cols
    ]

    static_inputs = (
        cfg["nn"]["static_inputs"]
        + element_inputs
    )

    state_inputs = cfg["nn"]["state"]

    return static_inputs + state_inputs


def report_nans(train_scaled, input_cols, cfg): 
    static_cols = [col for col in input_cols if col not in cfg["nn"]["state"]]
    scaled_static_cols = [f"{col}_scaled" for col in static_cols]

    nan_counts = train_scaled[scaled_static_cols].isna().sum()
    print(nan_counts[nan_counts > 0])


def prepare_data(data_dir, cfg, outdir: Optional[Path]=None):
    train, val, test = load_dataframes(data_dir)
    imputer = Imputer(cfg["data"]["fill_values"])
    train = imputer.fit_transform(train)

    missing = train[train["mass_catalyst"].isna()]
    print("Missing Catalyst Mass")
    print("Rows:", len(missing))
    print("Experiments:", missing["_id_experiment"].nunique())
    print("Materials:", missing["material_id"].nunique())
    train = train[~(train["mass_catalyst"].isna())]

    val = imputer.transform(val)
    test = imputer.transform(test)


    input_cols = derive_input_cols(train, cfg)
    cfg["nn"]["input_dim"] = len(input_cols)
    cfg["nn"]["input_cols"] = input_cols

    target_scales = calculate_target_scales(train=train, cfg=cfg, method="std")
    cfg["loss"]["target_scales"] = target_scales

    scaler = Scaler(input_cols)
    scaler.fit(train)
    train_scaled = scaler.scale(train)
    val_scaled = scaler.scale(val)
    test_scaled = scaler.scale(test)

    report_nans(train_scaled, input_cols, cfg)

    train_dataset = ExperimentDataset(train_scaled, cfg)
    val_dataset = ExperimentDataset(val_scaled, cfg)
    test_dataset = ExperimentDataset(test_scaled, cfg)

    if outdir:
        imputer.save(outdir / "imputer.pkl")
        scaler.save(outdir / "scaler.pkl")

    return {
        "train": train_dataset,
        "eval": val_dataset,
        "test": test_dataset,
        "scaler": scaler,
        "input_cols": input_cols
    }


def calculate_target_scales(
    train: pd.DataFrame,
    cfg,
    method="std",
    min_scale=1e-8,
):
    target_scales = {}

    for target in cfg["physics"]["targets"]:
        if target not in train.columns:
            raise ValueError(
                f"Target '{target}' not found in training dataframe."
            )

        values = pd.to_numeric(
            train[target],
            errors="coerce",
        ).dropna()

        if len(values) == 0:
            raise ValueError(
                f"No finite training values available for target '{target}'."
            )

        if method == "std":
            scale = values.std(ddof=0)

        elif method == "iqr":
            scale = (
                values.quantile(0.75)
                - values.quantile(0.25)
            )

        elif method == "range":
            scale = values.max() - values.min()

        else:
            raise ValueError(
                f"Unknown target scale method: {method}"
            )

        if not pd.notna(scale) or scale < min_scale:
            scale = 1.0

        target_scales[target] = float(scale)

    return target_scales


def plot_loss_history(trainer: Trainer, outdir=None):
    loss_names = [
        "total",
        "carbon",
        "nitrogen",
        "mass_soot_remaining_mg",
        "soot_oxidation_co2_concentration_ppm",
        "soot_oxidation_co_concentration_ppm",
        "soot_oxidation_co2_selectivity",
        "nox_ppm",
        "no2_ppm",
        "no_ppm",
        "n2_ppm",
        "no2_fraction_of_nox",
    ]

    labels = {
        "total": "Total Loss",
        "carbon": "Carbon Loss",
        "nitrogen": "Nitrogen Loss",
        "mass_soot_remaining_mg": "Soot Mass",
        "soot_oxidation_co2_concentration_ppm": "CO₂",
        "soot_oxidation_co_concentration_ppm": "CO",
        "soot_oxidation_co2_selectivity": "S(CO₂)",
        "nox_ppm": "NOx",
        "no2_ppm": "NO₂",
        "no_ppm": "NO",
        "n2_ppm": "N₂",
        "no2_fraction_of_nox": "S(NO₂)",
    }

    n_cols = 4
    n_rows = math.ceil(len(loss_names) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
    axes = axes.flatten()

    for ax, name in zip(axes, loss_names):
        train_key = f"train_{name}"
        eval_key = f"eval_{name}"

        if train_key in trainer.history:
            epochs = range(1, len(trainer.history[train_key]) + 1)
            ax.plot(epochs, trainer.history[train_key], label="Train")

        if eval_key in trainer.history:
            epochs = range(1, len(trainer.history[eval_key]) + 1)
            ax.plot(epochs, trainer.history[eval_key], label="Eval")

        if trainer.best_epoch is not None:
            ax.axvline(
                trainer.best_epoch,
                color="red",
                linestyle="--",
                linewidth=1.2,
                label=f"Best epoch ({trainer.best_epoch})",
            )

        ax.set_title(labels[name])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    for ax in axes[len(loss_names):]:
        ax.remove()

    fig.tight_layout()

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / "loss_history.png", dpi=300, bbox_inches="tight")

    return fig


def check_experiments_run(model, train_dataset):
    model.eval()

    for i in range(len(train_dataset)):
        experiment = train_dataset[i]

        try:
            with torch.no_grad():
                pred = model(experiment)

            if not all(
                torch.isfinite(v).all()
                for v in pred.values()
                if torch.is_tensor(v)
            ):
                print(f"Non-finite outputs in experiment {i}: {experiment['experiment_id']}")
                break

        except Exception as e:
            print(f"Failed experiment {i}: {experiment['experiment_id']}")
            print(e)
            break


def load_model(model_dir, just_model=True):
    models_module = importlib.import_module("src.architectures")

    with open(model_dir / "config.json", "r") as f:
        cfg = json.load(f)
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    model_class = getattr(models_module, cfg["model_name"])
    model = model_class(cfg, cfg["nn"]["input_dim"], scaler)
    
    state_dict = torch.load(model_dir / "best_model.pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    if just_model:
        return model
    else:
        return {
            "model": model,
            "config": cfg,
            "scaler": scaler
        }
