comet_support = True
try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False
import os
import time
import warnings
import argparse

import pandas as pd
import numpy as np
import lightning as L
from torch.nn import CrossEntropyLoss

from utils import seed_everything, create_logger, create_file, save_output, sort_file, plot_umap, calculate_cluster_metrics
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
    if cfg.RESULT.SAVE_MODEL:
        model_dir = os.path.join(cfg.RESULT.OUTPUT_DIR, cfg.DATASET.NAME, cfg.RESULT.SAVE_MODEL_DIR)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    cluster_metric_dir = os.path.join(cfg.RESULT.OUTPUT_DIR, cfg.DATASET.NAME, cfg.RESULT.SAVE_CLUSTER_DIR)
    if not os.path.exists(cluster_metric_dir):
        os.makedirs(cluster_metric_dir)
    lightning_dir = os.path.join(cfg.RESULT.OUTPUT_DIR, cfg.RESULT.LIGHTNING_LOG_DIR)
    if not os.path.exists(lightning_dir):
        os.makedirs(lightning_dir)
    plot_dir = os.path.join(cfg.RESULT.OUTPUT_DIR, cfg.DATASET.NAME, cfg.RESULT.SAVE_PLOT_DIR)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if cfg.RESULT.SAVE_RESULT:
        output_file = os.path.join(cfg.RESULT.OUTPUT_DIR, cfg.DATASET.NAME, f'MAGNET_{cfg.DATASET.NAME}_cls.csv')
        sorted_output_file = os.path.join(cfg.RESULT.OUTPUT_DIR, cfg.DATASET.NAME, f'MAGNET_{cfg.DATASET.NAME}_cls_sorted.csv')
        output_file_time = os.path.join(cfg.RESULT.OUTPUT_DIR, cfg.DATASET.NAME, f'MAGNET_{cfg.DATASET.NAME}_time.csv')
        create_file(file_dir=output_file, header=cfg.RESULT.FILE_HEADER_CLF)
        create_file(file_dir=output_file_time, header=cfg.RESULT.FILE_HEADER_TIME)

    split_folder = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.SPLITS)

    cls_results = {}
    time_results = []

    for seed in cfg.DATASET.SEEDS:
        seed_everything(seed)

        comet_logger = create_logger(config=cfg, name=f"test_{cfg.DATASET.NAME}_{seed}", style="comet",
                                     use=cfg.COMET.USE, support=comet_support)
        csv_logger = create_logger(config=cfg, name=f"test_{cfg.DATASET.NAME}_{seed}", style="csv",
                                   use=cfg.CSV.USE, support=True)
        loggers = [logger for logger in [comet_logger, csv_logger] if logger is not None]

        start_time = time.time()
        print(f"==> Loading data from seed {seed}...")
        multiomics = MultiomicsDataset(
            split_folder=split_folder,
            dataset_name=cfg.DATASET.NAME,
            modalities=cfg.DATASET.OMICS,
            seed=seed,
            num_classes=cfg.DATASET.NUM_CLASSES,
            class_names=cfg.DATASET.CLASS_NAMES,
            sparsity_rate=cfg.DATASET.SPARSITY_RATES,
            tune_hyperparameters=False)


        print("\n   ==> Building model...")
        modality_features = [
            multiomics.get_data(omics_idx=modality_idx).shape[1]
            for modality_idx in range(len(cfg.DATASET.OMICS))
        ]

        encoders = [
            MLPEncoder(
                in_dims=modality_features[modality_idx],
                hid_dims=cfg.ENCODER.HID_DIMS,
                dropout_rate=cfg.ENCODER.DROPOUT_RATE,
            )
            for modality_idx in range(len(cfg.DATASET.OMICS))
        ]

        attention_layer = MultiHeadAttentionLayer(hid_dims=cfg.ENCODER.HID_DIMS, num_heads=cfg.ATTENTION.NUM_HEADS)

        decoder = GNNDecoder(
            hid_dims=cfg.DECODER.HID_DIMS,
            out_dims=cfg.DATASET.NUM_CLASSES,
            num_layers=cfg.DECODER.NUM_LAYERS,
            dropout_rate=cfg.DECODER.DROPOUT_RATE,
            negative_slope=cfg.DECODER.NEGATIVE_SLOPE
        )
        loss_fn = CrossEntropyLoss()

        model = MAGNETTrainer(
            dataset=multiomics,
            unimodal_encoders=encoders,
            attention_layer=attention_layer,
            gnn_decoder=decoder,
            loss_fn=loss_fn,
            lr=cfg.SOLVER.LR,
            wd=cfg.SOLVER.WD,
        )

        trainer = L.Trainer(
            max_epochs=cfg.SOLVER.MAX_EPOCHS,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=1,
            logger=loggers,
            default_root_dir=lightning_dir
        )

        # ---- setup training model and trainer ----
        print("\n   ==> Training model...")
        trainer.fit(model)

        end_time = time.time()

        if cfg.RESULT.SAVE_MODEL:
            save_model_name = cfg.RESULT.SAVE_MODEL_TMPL.format(dataset_name=cfg.DATASET.NAME, seed=seed)
            trainer.save_checkpoint(os.path.join(model_dir, save_model_name))

        print("\n   ==> Testing model...")
        trainer.test(model)

        running_time = end_time - start_time
        time_results.append(running_time)
        if cfg.RESULT.SAVE_RESULT:
            time_result = [seed, running_time]
            save_output(output_file_time, time_result)

        for metric_key, metric_value in trainer.logged_metrics.items():
            cls_results.setdefault(metric_key.replace("test_", ""), []).append(metric_value.item())

            if cfg.RESULT.SAVE_RESULT:
                result = [metric_key.replace("test_", ""), seed, metric_value.item()]
                save_output(output_file, result)

        train_embeddings, train_labels, test_embeddings, test_labels = model.get_embeddings()

        save_path_train_umap = os.path.join(plot_dir, cfg.RESULT.SAVE_PLOT_TMPL.format(name="UMAP", mode="train", title="{title}",seed=seed, format="{format}"))
        save_path_test_umap = os.path.join(plot_dir, cfg.RESULT.SAVE_PLOT_TMPL.format(name="UMAP", mode="test", title="{title}", seed=seed, format="{format}"))

        plot_umap(cfg, train_embeddings, train_labels, cfg.DATASET.CLASS_NAMES, save_path_train_umap, seed)
        plot_umap(cfg, test_embeddings, test_labels, cfg.DATASET.CLASS_NAMES, save_path_test_umap, seed)
        train_cluster_metrics = calculate_cluster_metrics(train_embeddings, train_labels)
        test_cluster_metrics =  calculate_cluster_metrics(test_embeddings, test_labels)
        train_metrics_df = pd.DataFrame.from_dict(train_cluster_metrics, orient="index")
        test_metrics_df = pd.DataFrame.from_dict(test_cluster_metrics, orient="index")
        save_path_train_cluster = os.path.join(cluster_metric_dir, cfg.RESULT.SAVE_CLUSTER_TMPL.format(mode="train", seed=seed))
        save_path_test_cluster = os.path.join(cluster_metric_dir, cfg.RESULT.SAVE_CLUSTER_TMPL.format(mode="test", seed=seed))
        train_metrics_df.to_csv(save_path_train_cluster)
        test_metrics_df.to_csv(save_path_test_cluster)

    if cfg.RESULT.SAVE_RESULT:
        sort_file(output_file, sorted_output_file, by=cfg.RESULT.FILE_HEADER_CLF[0:2])

    print(f"\n==> Showing results...")
    exe_time = round(np.mean(time_results))
    print(f"Execution time: {exe_time} seconds")
    for metric_name, values in cls_results.items():
        average = np.mean(values)
        std = np.std(values)
        print(f"    - {metric_name}: {average:.3f}±{std:.3f}")


if __name__ == '__main__':
    main()
