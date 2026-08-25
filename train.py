from pathlib import Path
from src.architectures import NOxSelectivityModel, ArrheniusNOxSelectivityModel
from src.trainer import Trainer
from src.utils import *
from src.analyser import ModelAnalyser

split = "experiment_split"
seed = 1
data_dir = Path(f"data/{split}/{seed}")
CFG = {
    "data": {
        "fill_values": {
            "sup_calcination_temp": "mean",
            "sup_calcination_time": "median",
            "start_temp_K": 298
        },
        "ode_start": "first_data_point"  # "experiment_start" or "first_data_point" - controls whether the model will integrate from the actual start to first measured or not. 
    },
    "nn": {
        "hidden_dim": [16, 16],
        "output_dim": 6,
        "activation": "ReLU",
        "static_inputs": ["NO_initial_ppm", "NO2_initial_ppm", "O2_vol%", "mass_catalyst", "tight_contact", "mass_silica_beads", "gas_flow_ml_min", "sup_calcination_temp", "sup_calcination_time", "Sbet"], # elements automatically included
        "state": ["mass_soot_remaining_mg", "temperature_K"],
        "oxidisable_mass_fraction_bounds": [0.8, 1.3] # f_min and f_max (fraction of soot that is oxidisable to CO or CO2)
    },
    "physics": {
        "required_unscaled_inputs": ["mass_soot_remaining_mg", "ramp_rate_K_min", "temperature_K", "O2_fraction"],
        "targets": ["no2_fraction_of_nox", "nox_ppm", "no2_ppm", "no_ppm", "n2_ppm", "mass_soot_remaining_mg", "soot_oxidation_co2_concentration_ppm", "soot_oxidation_co_concentration_ppm", "soot_oxidation_co2_selectivity"],
        "rate_scales": {
            "k1": 1e-6,
            "k2": 1.0,
            "k5": 1.0
        },
        "solver": {
            "method": "rk4",
            "options": {"step_size": 0.1}
        },
        "arrhenius": {
            "T_ref_K": 700.0,
            "Ea_scale_kJ_mol": 50.0,
            "log_kref_multiplier_bound": 2.0,
            "state_correction_limit": 0.5
        },
    },
    "loss": {
        "target_weights": {}, # for each loss contribution with N or C balance
        "target_scales": {}, # calculate this as std. of targets in train set - used to balance loss contributions
        "carbon_weight": 0.5, # in case want to prioritize loss contributions based on C
        "nitrogen_weight": 0.5
    },
    "train": {
        "batch_size": 32, # experiments per batch
        "epochs": 1000,
        "early_stopping_patience": 25,
        "optimiser_name": "AdamW",
        "optimiser_kwargs": {
            "lr": 4e-4,
            "weight_decay": 1e-6
        },
        "debug": {
            "enabled": True,
            "print_top_experiments": 5,
            "print_gradients": True,
            "gradient_clip_norm": None
        }
    }
}

if __name__ == "__main__":
    cfg = CFG.copy()
    outdir = Path(f"models/arrhenius_like/{split}/{seed}")
    outdir.mkdir(exist_ok=True, parents=True)

    data = prepare_data(data_dir, cfg, outdir)
    torch.save(data["test"], outdir / "test_dataset.pt")
    model = NOxSelectivityModel(cfg, len(data["input_cols"]), data["scaler"])
    # check_experiments_run(model, data["train"])

    cfg["model_name"] = model.__class__.__name__
    cfg["nn"]["total_learnable_parameters"] = model.count_parameters()

    with open(outdir / "config.json", "w") as f:
        json.dump(cfg, f, indent=4)

    trainer = Trainer(cfg, model, data["train"], data["eval"], output_dir=outdir)
    trainer.train()
    plot_loss_history(trainer, outdir=outdir)
    analyser = ModelAnalyser(model, data["test"], outdir = Path(outdir / "analysis"))
    analyser.analyse()