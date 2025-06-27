import os

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

from utils import load_dataset, load_label, align_modalities, cosine_distance


class MultiomicsDataset:
    def __init__(
            self,
            split_folder: str,
            dataset_name: str,
            modalities: list[str],
            seed: int,
            num_classes: int,
            class_names: list[str],
            sparsity_rate: float,
            tune_hyperparameters: bool = False
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_folder = os.path.join(split_folder, dataset_name, str(seed))
        self.modalities = modalities
        self._num_omics = len(self.modalities)
        self._num_classes = num_classes
        self.class_names = class_names
        self.sparsity_rate = sparsity_rate
        self.tune_hyperparameters = tune_hyperparameters

        # load labels
        self.labels, self.train_mask, self.val_mask, self.test_mask, self.test_mask_paired, self.test_mask_unpaired = self._load_labels()
        self.data, self.data_indices, self.modality_mask = self._load_data(self.labels)

        # build patient similarity network
        self.adj_mat = self._build_patient_sim_net()

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index):
        return self.data, self.data_indices, self.modality_mask, self.labels

    @property
    def num_omics(self) -> int:
        return self._num_omics

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def _load_labels(self):
        train_paired, val_paired, test_paired = load_label(self.dataset_folder, "paired")
        train_unpaired, val_unpaired, test_unpaired = load_label(self.dataset_folder, "unpaired")

        train_labels = pd.concat([train_paired, train_unpaired]).sort_index()
        test_labels = pd.concat([test_paired, test_unpaired]).sort_index()

        if self.tune_hyperparameters:
            val_labels = pd.concat([val_paired, val_unpaired]).sort_index()
            labels = pd.concat([train_labels, val_labels, test_labels])
        else:
            train_labels = pd.concat([train_labels, val_paired, val_unpaired]).sort_index()
            val_labels = None
            labels = pd.concat([train_labels, test_labels])

        num_patients = len(labels)
        train_mask = torch.zeros(num_patients, dtype=torch.bool)
        test_mask = torch.zeros(num_patients, dtype=torch.bool)

        train_mask[:len(train_labels)] = True
        test_mask[-len(test_labels):] = True

        val_mask = None
        if self.tune_hyperparameters:
            val_mask = torch.zeros(num_patients, dtype=torch.bool)
            val_mask[len(train_labels):-len(test_labels)] = True

        test_mask_paired = torch.zeros(num_patients, dtype=torch.bool)
        test_mask_unpaired = torch.zeros(num_patients, dtype=torch.bool)

        for idx in test_paired.index:
            test_mask_paired[labels.index.get_loc(idx)] = True

        for idx in test_unpaired.index:
            test_mask_unpaired[labels.index.get_loc(idx)] = True

        return labels, train_mask, val_mask, test_mask, test_mask_paired, test_mask_unpaired

    def _load_data(self, labels: pd.DataFrame):
        modality_data, modality_data_indices, mask_matrix = [], [], []

        for modality_name in self.modalities:
            data = load_dataset(self.dataset_folder, modality_name, labels)
            data = data.reindex(labels.index).dropna()
            modality_data_indices.append(data.index)
            modality_data.append(torch.tensor(data.values, dtype=torch.float32))

            mask_column = labels.index.isin(data.index).astype(int)
            mask_matrix.append(mask_column)

        modality_mask = torch.tensor(np.array(mask_matrix), dtype=torch.float32).T

        return modality_data, modality_data_indices, modality_mask

    def get_modality_mask(self):
        return self.modality_mask

    def get_data(self, omics_idx: int = None):
        if omics_idx is not None:
            return self.data[omics_idx]

        return self.data

    def get_labels(self):
        return self.labels

    def _build_patient_sim_net(self):
        aligned_modalities = align_modalities(self.data, self.data_indices, self.labels)
        patient_rep = torch.cat(aligned_modalities, dim=1)

        # Use modality_mask to indicate available modalities for each patient
        overall_mask = self._build_overall_mask()
        adj_mat = self._compute_adj_mat(patient_rep, overall_mask)
        adj_mat = self._apply_sparsity(adj_mat, self.train_mask)
        adj_mat = self._apply_sparsity(adj_mat, self.test_mask)

        if self.tune_hyperparameters:
            adj_mat = self._apply_sparsity(adj_mat, self.val_mask)

        return adj_mat

    def _apply_sparsity(self, adj_mat, mask):
        indices = mask.nonzero(as_tuple=False).squeeze()
        sub_adj = adj_mat[indices][:, indices]
        sorted_sims = sub_adj.flatten().sort(descending=False)

        num_edges = len(sorted_sims.values)
        threshold_idx = int(self.sparsity_rate * num_edges)
        threshold_val = sorted_sims.values[threshold_idx]

        sub_adj[sub_adj < threshold_val] = 0.0

        # Ensure at least one edge per node (not self-loop)
        for i in range(sub_adj.size(0)):
            if (sub_adj[i] == 0).all():
                # Find the most similar neighbor (excluding self)
                row = adj_mat[indices[i]]
                row[indices[i]] = 0.0
                top_idx = torch.argmax(row[indices])  # highest among masked nodes
                sub_adj[i, top_idx] = row[indices[top_idx]]
                sub_adj[top_idx, i] = row[indices[top_idx]]  # keep symmetric if undirected

        for i, row in enumerate(indices):
            for j, col in enumerate(indices):
                adj_mat[row, col] = sub_adj[i, j]

        return adj_mat

    def _build_overall_mask(self):
        overall_mask = []
        for modality_idx in range(self.num_omics):
            modality_feature_dim = self.data[modality_idx].shape[1]
            expanded_mask = self.modality_mask[:, modality_idx].unsqueeze(-1).repeat(1, modality_feature_dim)
            overall_mask.append(expanded_mask)

        overall_mask = torch.cat(overall_mask, dim=1)

        return overall_mask

    def _compute_adj_mat(self, patient_rep, overall_mask):
        num_patients = len(self.labels)
        adj_mat = torch.zeros((num_patients, num_patients), dtype=torch.float32)

        for i in range(num_patients):
            for j in range(i, num_patients):
                if i == j:
                    adj_mat[i, j] = 0.0
                else:
                    shared_mask = overall_mask[i] * overall_mask[j]
                    if shared_mask.sum() > 0:
                        similarity = cosine_distance(patient_rep[i], patient_rep[j], shared_mask)
                        adj_mat[i, j] = adj_mat[j, i] = abs(similarity)
                    else:
                        adj_mat[i, j] = adj_mat[j, i] = 0.0

        adj_mat = self._remove_cross_set_edges(adj_mat)

        return adj_mat

    def _remove_cross_set_edges(self, adj_mat):
        num_patients = len(self.labels)

        for i in range(num_patients):
            for j in range(num_patients):
                if i != j:
                    if (self.test_mask[i] and not self.test_mask[j]) or (self.test_mask[j] and not self.test_mask[i]):
                        adj_mat[i, j] = adj_mat[j, i] = 0.0

                    if self.tune_hyperparameters:
                        if (self.train_mask[i] and self.val_mask[j]) or (self.train_mask[j] and self.val_mask[i]):
                            adj_mat[i, j] = adj_mat[j, i] = 0.0

        return adj_mat

    def build_graph_data(self, embedding, mode):
        adj_mat = self.adj_mat.clone().to(embedding.device)

        if mode == 'train':
            adj_mat[self.test_mask, :] = 0
            adj_mat[:, self.test_mask] = 0
            if self.tune_hyperparameters:
                adj_mat[self.val_mask, :] = 0
                adj_mat[:, self.val_mask] = 0
        elif mode == 'val':
            adj_mat[self.train_mask, :] = 0
            adj_mat[:, self.train_mask] = 0
            adj_mat[self.test_mask, :] = 0
            adj_mat[:, self.test_mask] = 0
        elif mode == 'test':
            adj_mat[self.train_mask, :] = 0
            adj_mat[:, self.train_mask] = 0
        else:
            raise ValueError("Mode must be either 'train', 'val', or 'test'.")

        edge_index = (adj_mat > 0).nonzero(as_tuple=False).T
        edge_attr_adj = adj_mat[edge_index[0], edge_index[1]]

        # Edge mask for KL divergence
        N = embedding.size(0)
        edge_mask = torch.zeros(N, N, device=embedding.device)
        edge_mask[edge_index[0], edge_index[1]] = 1.0

        data = Data(
            x=embedding,
            y=torch.tensor(self.labels.values, dtype=torch.long, device=embedding.device).squeeze(),
            edge_index=edge_index,
            edge_attr=edge_attr_adj,
            train_mask=self.train_mask,
            val_mask=self.val_mask,
            test_mask=self.test_mask,
            test_mask_paired=self.test_mask_paired,
            test_mask_unpaired=self.test_mask_unpaired
        )

        return data, adj_mat, edge_mask
