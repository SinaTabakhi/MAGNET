comet_support = True
try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False

import os
import warnings
import argparse
from pathlib import Path

import lightning as L
import torch
from torch.nn import CrossEntropyLoss
import ray
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler

from utils import seed_everything, create_logger
from dataloader.dataloader import MultiomicsDataset
from model.base_models import MLPEncoder, GNNDecoder, MultiHeadAttentionLayer
from trainer.trainer import MAGNETTrainer
from configs.config import get_cfg_defaults


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="MAGNET for multiomics data integration")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    args = parser.parse_args()

    return args

def trial_dirname_creator(trial):
    return f"trial_{trial.trial_id}"  # Use only the trial ID to simplify names

def hypertune_magnet(hyper_cfg, setup_cfg, split_folder, lightning_dir):
    original_cwd = os.getcwd()  # Save the original working directory
    script_base_dir = os.path.dirname(os.path.abspath(__file__))  # Script's base directory
    os.chdir(script_base_dir)  # Set working directory to script's base directory

    try:
        accuracies = []

        for seed in setup_cfg.DATASET.SEEDS:
            seed_everything(seed)
            comet_logger = create_logger(config=setup_cfg, name=f"tune_{setup_cfg.DATASET.NAME}_{seed}", style="comet",
                                         use=setup_cfg.COMET.USE, support=comet_support, hyper_params=hyper_cfg)
            csv_logger = create_logger(config=setup_cfg, name=f"tune_{setup_cfg.DATASET.NAME}_{seed}", style="csv",
                                       use=setup_cfg.CSV.USE, support=True, hyper_params=hyper_cfg)
            loggers = [logger for logger in [comet_logger, csv_logger] if logger is not None]

            print(f"==> Loading data from seed {seed}...")
            multiomics = MultiomicsDataset(
                split_folder=split_folder,
                dataset_name=setup_cfg.DATASET.NAME,
                modalities=setup_cfg.DATASET.OMICS,
                seed=seed,
                num_classes=setup_cfg.DATASET.NUM_CLASSES,
                class_names=setup_cfg.DATASET.CLASS_NAMES,
                sparsity_rate=hyper_cfg["sparsity_rate"],
                tune_hyperparameters=setup_cfg.TUNE.USE)

            print("\n   ==> Building model...")
            modality_features = [
                multiomics.get_data(omics_idx=modality_idx).shape[1]
                for modality_idx in range(len(setup_cfg.DATASET.OMICS))
            ]

            encoders = [
                MLPEncoder(
                    in_dims=modality_features[modality_idx],
                    hid_dims=hyper_cfg["hid_dims"],
                    dropout_rate=hyper_cfg["dropout_rate"],
                )
                for modality_idx in range(len(setup_cfg.DATASET.OMICS))
            ]

            attention_layer = MultiHeadAttentionLayer(hid_dims=hyper_cfg["hid_dims"], num_heads=hyper_cfg["att_num_heads"])

            decoder = GNNDecoder(
                hid_dims=hyper_cfg["hid_dims"],
                out_dims=setup_cfg.DATASET.NUM_CLASSES,
                num_layers=hyper_cfg["gnn_num_layers"],
                dropout_rate=hyper_cfg["dropout_rate"],
                negative_slope=setup_cfg.DECODER.NEGATIVE_SLOPE,
            )
            loss_fn = CrossEntropyLoss()

            model = MAGNETTrainer(
                dataset=multiomics,
                unimodal_encoders=encoders,
                attention_layer=attention_layer,
                gnn_decoder=decoder,
                loss_fn=loss_fn,
                lr=hyper_cfg["lr"],
                wd=setup_cfg.SOLVER.WD,
            )

            trainer = L.Trainer(
                max_epochs=setup_cfg.TUNE.MAX_EPOCHS,
                accelerator="auto",
                devices="auto",
                log_every_n_steps=1,
                logger=loggers,
                default_root_dir=lightning_dir,
            )

            print("\n   ==> Training model...")
            trainer.fit(model)

            accuracy = trainer.logged_metrics[setup_cfg.TUNE.METRIC]

            if isinstance(accuracy, torch.Tensor):
                accuracy = accuracy.item()
            accuracies.append(accuracy)
            print(f"Seed {seed} - Accuracy: {accuracy} - Config: {hyper_cfg}")

        avg_accuracy = sum(accuracies) / len(accuracies)
        session.report({"accuracy": avg_accuracy})

    finally:
        os.chdir(original_cwd)

def main():
    warnings.filterwarnings(action="ignore")

    # ---- setup configs ----
    args = arg_parse()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    # ---- setup folders and paths ----
    if not os.path.exists(cfg.RESULT.OUTPUT_DIR):
        os.makedirs(cfg.RESULT.OUTPUT_DIR)
    if comet_support and cfg.COMET.USE:
        comet_dir = os.path.join(cfg.RESULT.OUTPUT_DIR, cfg.COMET.LOG_DIR)
        if not os.path.exists(comet_dir):
            os.makedirs(comet_dir)
    if cfg.CSV.USE:
        csv_dir = os.path.join(cfg.RESULT.OUTPUT_DIR, cfg.CSV.LOG_DIR)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
    if cfg.TUNE.USE:
        output_hyper = os.path.join(cfg.RESULT.OUTPUT_DIR, f'MAGNET_hyper_{cfg.DATASET.NAME}.csv')
        tune_dir = os.path.join(cfg.RESULT.OUTPUT_DIR, cfg.TUNE.LOG_DIR)
        if not os.path.exists(tune_dir):
            os.makedirs(tune_dir)
    lightning_dir = os.path.join(cfg.RESULT.OUTPUT_DIR, cfg.RESULT.LIGHTNING_LOG_DIR)
    if not os.path.exists(lightning_dir):
        os.makedirs(lightning_dir)

    split_folder = Path(cfg.DATASET.ROOT) / cfg.DATASET.SPLITS
    absolute_path = Path(tune_dir).resolve()
    storage_path = absolute_path.as_uri()

    search_space = {
        "hid_dims": tune.grid_search([128, 256]),
        "sparsity_rate": tune.choice([round(x, 2) for x in torch.arange(0.5, 1.0, 0.05).tolist()]),
        "dropout_rate": tune.choice([0.0, 0.1, 0.2, 0.3]),
        "att_num_heads": tune.grid_search([2, 4, 8]),
        "gnn_num_layers": tune.grid_search([2]),
        "lr": tune.loguniform(1e-5, 1e-3)
    }

    ray.init()
    resources = ray.available_resources()
    num_gpus = int(resources.get("GPU", 0))

    if num_gpus > 0:
        resources_per_trial = {"gpu": num_gpus}
    else:
        num_cpus = int(resources.get("CPU", 0))
        resources_per_trial = {"cpu":  max(1, num_cpus // 2)}

    scheduler = ASHAScheduler(max_t=cfg.TUNE.MAX_EPOCHS, grace_period=cfg.TUNE.MIN_EPOCHS, reduction_factor=2, metric="accuracy", mode="max")

    analysis = tune.run(
        tune.with_parameters(hypertune_magnet, setup_cfg=cfg, split_folder=split_folder, lightning_dir=lightning_dir),
        config=search_space,
        scheduler=scheduler,
        num_samples=cfg.TUNE.NUM_SAMPLES,
        resources_per_trial=resources_per_trial,
        trial_dirname_creator=trial_dirname_creator,
        storage_path=storage_path,
    )

    best_trial_in_run = analysis.get_best_trial(metric="accuracy", mode="max", scope="all")
    print(f"Best result ever was achieved by trial {best_trial_in_run}")

    df = analysis.dataframe("accuracy", "max")
    df.to_csv(output_hyper, index=False)


if __name__ == '__main__':
    main()
