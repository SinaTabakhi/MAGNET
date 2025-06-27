import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import csv
import random

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.metrics import silhouette_score, davies_bouldin_score
from lightning.pytorch.loggers import CometLogger, CSVLogger
import umap
import matplotlib.pyplot as plt


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def create_file(file_dir, header):
    with open(file_dir, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)


def save_output(file_dir, output, multi_rows=False):
    with open(file_dir, mode='a', newline='') as file:
        writer = csv.writer(file)
        if multi_rows:
            writer.writerows(output)
        else:
            writer.writerow(output)


def sort_file(origin_dir, sorted_dir, by):
    df = pd.read_csv(origin_dir)
    df = df.sort_values(by=by)
    df.to_csv(sorted_dir, index=False)


def load_label(dataset_folder: str, split_type: str):
    train_labels_path = os.path.join(dataset_folder, f"label_train_{split_type}.csv")
    val_labels_path = os.path.join(dataset_folder, f"label_val_{split_type}.csv")
    test_labels_path = os.path.join(dataset_folder, f"label_test_{split_type}.csv")

    train_labels = pd.read_csv(train_labels_path, index_col=0)
    val_labels = pd.read_csv(val_labels_path, index_col=0)
    test_labels = pd.read_csv(test_labels_path, index_col=0)

    for df in [train_labels, val_labels, test_labels]:
        if df.empty and "Class" in df.columns:
            df["Class"] = df["Class"].astype("int64")

    return train_labels, val_labels, test_labels


def load_dataset(dataset_folder: str, modality_name: str, labels: pd.DataFrame):
    unimodal_path = os.path.join(dataset_folder, f"{modality_name}.csv")
    unimodal_data = pd.read_csv(unimodal_path, index_col=0)

    common_indices = labels.index.intersection(unimodal_data.index)
    unimodal_data = unimodal_data.loc[common_indices].sort_index()

    return unimodal_data


def align_modalities(data, data_indices, label):
    aligned_modalities = []

    # Align each data modality
    for modality_idx, data_modality in enumerate(data):
        modality_indices = data_indices[modality_idx]

        aligned_data = torch.zeros((len(label.index), data_modality.shape[1]),
                                   dtype=data_modality.dtype, device=data_modality.device)
        index_mapping = {patient_id: idx for idx, patient_id in enumerate(modality_indices)}

        for patient_row, patient_id in enumerate(label.index):
            if patient_id in index_mapping:
                aligned_data[patient_row] = data_modality[index_mapping[patient_id]]

        aligned_modalities.append(aligned_data)

    return aligned_modalities


def cosine_distance(x1, x2, mask=None, eps=1e-8):
    if mask is not None:
        x1 = x1 * mask
        x2 = x2 * mask

    norm_i = torch.norm(x1, p=2)
    norm_j = torch.norm(x2, p=2)
    similarity = torch.dot(x1, x2) / (norm_i * norm_j + eps)  # Avoid division by zero
    return similarity


def evaluate_classification_performance(y_true, probs, num_classes):
    y_pred = torch.argmax(probs, dim=-1)
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    if num_classes > 2:
        acc = accuracy_score(y_true_np, y_pred_np)
        f1_micro = f1_score(y_true_np, y_pred_np, average="micro")
        f1_macro = f1_score(y_true_np, y_pred_np, average="macro")
        f1_weighted = f1_score(y_true_np, y_pred_np, average="weighted")
        mcc = matthews_corrcoef(y_true_np, y_pred_np)

        metrics = {
            'acc': acc,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'mcc': mcc
        }

    else:
        probs_np = probs[:, 1].detach().cpu().numpy()
        acc = accuracy_score(y_true_np, y_pred_np)
        f1 = f1_score(y_true_np, y_pred_np)
        mcc = matthews_corrcoef(y_true_np, y_pred_np)

        if len(np.unique(y_true_np)) == 1:
            auroc = None
            auprc = None
        else:
            auroc = roc_auc_score(y_true_np, probs_np)
            auprc = average_precision_score(y_true_np, probs_np)

        metrics = {
            'acc': acc,
            'auroc': auroc,
            'auprc': auprc,
            'f1': f1,
            'mcc': mcc,
        }

    return metrics


def create_logger(config, name, style, use, support=True, hyper_params=None):
    if style == "comet":
        if use and support:
            save_dir = os.path.join(config.RESULT.OUTPUT_DIR, config.COMET.LOG_DIR)
            comet_logger = CometLogger(
                project_name=config.COMET.PROJECT_NAME,
                workspace=config.COMET.WORKSPACE,
                save_dir=save_dir,
                experiment_name=f"{name}",
                auto_output_logging="simple",
                log_graph=True,
                log_code=False,
                log_git_metadata=False,
                log_git_patch=False,
                auto_param_logging=False,
                auto_metric_logging=False
            )
            if hyper_params is None:
                hyper_params = {
                    "sparsity_rate": config.DATASET.SPARSITY_RATES,
                    "hid_dims": config.ENCODER.HID_DIMS,
                    "mlp_dropout_rate": config.ENCODER.DROPOUT_RATE,
                    "gat_num_layers": config.DECODER.NUM_LAYERS,
                    "gat_dropout_rate": config.DECODER.DROPOUT_RATE,
                    "gat_negative_slope": config.DECODER.NEGATIVE_SLOPE,
                    "lr": config.SOLVER.LR,
                    "wd": config.SOLVER.WD,
                    "max_epochs": config.SOLVER.MAX_EPOCHS
                }
            comet_logger.experiment.log_parameters(hyper_params)
            return comet_logger
    elif style == "csv":
        if use:
            save_dir = os.path.join(config.RESULT.OUTPUT_DIR, config.CSV.LOG_DIR)
            csv_logger = CSVLogger(save_dir=save_dir, name=f"{name}")
            return csv_logger
    return None


def plot_umap(config, embeddings_to_plot, labels_to_plot, class_names, save_path, seed):
    all_values = np.concatenate([np.array(values) for values in labels_to_plot.values()])

    classes = np.unique(all_values)

    if len(class_names) == 2:
        colormap = plt.cm.get_cmap('Paired', len(class_names))
    else:
        colormap = plt.cm.get_cmap('tab10', len(class_names))

    for title, embeddings in embeddings_to_plot.items():
        labels = labels_to_plot[title]

        colors = [colormap(label) for label in labels]

        umap_reducer = umap.UMAP(n_neighbors=config.UMAP.N_NEIGHBORS, min_dist=config.UMAP.MIN_DIST, n_components=2, random_state=seed)
        umap_embeddings = umap_reducer.fit_transform(embeddings)

        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colormap(class_id),
                              markersize=10, label=class_names[class_id]) for class_id in classes]

        fig, ax = plt.subplots(figsize=(7, 6))
        scatter = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=colors, s=20)
        ax.set_title(f"{title}")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.legend(handles=handles, title='Patient type', loc='best', edgecolor='black')

        save_path_pdf = save_path.format(title=title, format='pdf')
        save_path_svg = save_path.format(title=title, format='svg')
        plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
        plt.savefig(save_path_svg, format='svg', bbox_inches='tight')

        # plt.show()


def calculate_cluster_metrics(embeddings, labels):
    results = {}

    for key in embeddings:
        silhouette = silhouette_score(embeddings[key], labels[key])
        davies_bouldin = davies_bouldin_score(embeddings[key], labels[key])

        results[key] = {
            "Silhouette Score": silhouette,
            "Davies-Bouldin Index": davies_bouldin
        }

    return results
