import os
import warnings
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from preprocess.feature_selection import FeatureSelectionMethod, select_features
from configs.config import get_cfg_defaults


def split_label(label, ratio, seed, seed_folder, split_type, save=True):
    train_label, temp_label = train_test_split(label, test_size=ratio[1] + ratio[2], stratify=label, random_state=seed)
    val_label, test_label = train_test_split(temp_label, test_size=ratio[2]/(ratio[1] + ratio[2]), stratify=temp_label, random_state=seed)

    if save:
        train_label.sort_index().to_csv(os.path.join(seed_folder, f"label_train_{split_type}.csv"), index=True)
        val_label.sort_index().to_csv(os.path.join(seed_folder, f"label_val_{split_type}.csv"), index=True)
        test_label.sort_index().to_csv(os.path.join(seed_folder, f"label_test_{split_type}.csv"), index=True)

    return train_label, val_label, test_label

def save_data(seed_folder, file_path, data, label):
    data_path = os.path.join(seed_folder, file_path)
    common_indices = label.index.intersection(data.index)
    data.loc[common_indices].sort_index().to_csv(data_path, index=True)

def print_split_info(split_name, df, label_column='Class'):
    print(f" - {split_name}: #Patients: {len(df)}")
    # print(df[label_column].value_counts())

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
    main_folder = os.path.join("", cfg.DATASET.ROOT, cfg.DATASET.SPLITS, cfg.DATASET.NAME)
    main_folder_raw = os.path.join("", cfg.DATASET.ROOT, cfg.DATASET.PROCESSED, cfg.DATASET.NAME)
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    raw_files_paths = [os.path.join(main_folder_raw, f"{modality}.csv") for modality in cfg.DATASET.OMICS]
    raw_label_path = os.path.join(main_folder_raw, f"ClinicalMatrix.csv")

    multimodal_data = []

    for idx, path in enumerate(raw_files_paths):
        unimodal_data = pd.read_csv(path, sep=',', index_col=0)
        multimodal_data.append(unimodal_data)

    label = pd.read_csv(raw_label_path, sep=',', index_col=0)

    # Find paired patients across all modalities
    paired_patient_ids = label.index
    for unimodal_data in multimodal_data:
        paired_patient_ids = paired_patient_ids.intersection(unimodal_data.index)

    # Split the labels into paired and unpaired patients
    paired_labels = label.loc[paired_patient_ids]

    unpaired_labels = label.drop(paired_patient_ids)

    for seed in cfg.DATASET.SEEDS:
        print(f"\n\n************* Splitting data with seed {seed} ****************")
        seed_folder = os.path.join(main_folder, str(seed))
        os.makedirs(seed_folder, exist_ok=True)

        train_paired_labels, val_paired_labels, test_paired_labels = split_label(
            paired_labels, ratio=cfg.DATASET.SPLIT_RATIO,
            seed=seed, seed_folder=seed_folder,
            split_type="paired", save=True)
        print("Matched/Paired patients:")
        print_split_info("Train labels", train_paired_labels)
        print_split_info("Validation labels", val_paired_labels)
        print_split_info("Test labels", test_paired_labels)

        train_unpaired_labels, val_unpaired_labels, test_unpaired_labels = split_label(
            unpaired_labels, ratio=cfg.DATASET.SPLIT_RATIO,
            seed=seed, seed_folder=seed_folder,
            split_type="unpaired", save=True)
        print("\nUnmatched/Unpaired patients:")
        print_split_info("Train labels", train_unpaired_labels)
        print_split_info("Validation labels", val_unpaired_labels)
        print_split_info("Test labels", test_unpaired_labels)

        combined_labels = pd.concat([train_paired_labels, val_paired_labels, train_unpaired_labels, val_unpaired_labels])
        if combined_labels.index.duplicated().any():
            raise Exception("There are duplicate indices in combined_labels.")

        print("\nOmic modality shapes after feature selection")
        for modality_idx in range(len(cfg.DATASET.OMICS)):
            unimodal_data = multimodal_data[modality_idx]
            if unimodal_data.shape[1] > cfg.DATASET.NUM_FEATURES:
                common_indices = combined_labels.index.intersection(unimodal_data.index)
                X_train = unimodal_data.loc[common_indices].sort_index()
                y_train = combined_labels.loc[common_indices].sort_index().values.ravel()

                if X_train.shape[0] != y_train.shape[0]:
                    raise Exception("X_train and y_train do not have the same number of samples.")

                if not X_train.index.equals(combined_labels.loc[common_indices].sort_index().index):
                    raise Exception("X_train and y_train do not have the same indices.")

                selected_features = select_features(FeatureSelectionMethod.ANOVA, X_train, y_train, n_features=cfg.DATASET.NUM_FEATURES)

                unimodal_data = unimodal_data.loc[:, selected_features]

            print(f"  - {cfg.DATASET.OMICS[modality_idx]} shape: {unimodal_data.shape}")

            output_path = os.path.join(seed_folder, f"{cfg.DATASET.OMICS[modality_idx]}.csv")
            unimodal_data.sort_index().to_csv(output_path, index=True)


if __name__ == '__main__':
    main()