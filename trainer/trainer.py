from typing import List

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import lightning as L

from model.base_models import MLPEncoder, MultiHeadAttentionLayer, GNNDecoder
from dataloader.dataloader import MultiomicsDataset
from utils import align_modalities, evaluate_classification_performance


class MAGNETTrainer(L.LightningModule):
    def __init__(
        self,
        dataset: MultiomicsDataset,
        unimodal_encoders: List[MLPEncoder],
        attention_layer: MultiHeadAttentionLayer,
        gnn_decoder: GNNDecoder,
        loss_fn: CrossEntropyLoss,
        lr: float,
        wd: float
    ):
        super().__init__()
        self.save_hyperparameters("lr", "wd")
        self.dataset = dataset
        self.unimodal_encoders = nn.ModuleList(unimodal_encoders)
        self.attention_layer = attention_layer
        self.gnn_decoder = gnn_decoder
        self.loss_fn = loss_fn
        self.lr = lr
        self.wd = wd

        self.train_embeddings_to_plot = {}
        self.train_labels_to_plot = {}
        self.test_embeddings_to_plot = {}
        self.test_labels_to_plot = {}

    def get_embeddings(self):
        return self.train_embeddings_to_plot, self.train_labels_to_plot, self.test_embeddings_to_plot, self.test_labels_to_plot

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.8)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def similarity_kl_loss(self, similarities, embeddings, mask=None, alpha=1.0, eps=1e-12):
        device = embeddings.device
        n = embeddings.size(0)

        # Compute squared distances (Euclidean distance)
        sum_sq = torch.sum(embeddings ** 2, dim=1)
        dist_sq = sum_sq.unsqueeze(1) + sum_sq.unsqueeze(0) - 2 * torch.matmul(embeddings, embeddings.T)

        # Student-t kernel similarity
        q_matrix = torch.pow(1 + dist_sq / alpha, -(alpha + 1) / 2)

        # Remove self-similarity
        q_matrix = q_matrix * (1 - torch.eye(n, device=device))

        # Apply mask if provided
        if mask is not None:
            q_matrix = q_matrix.masked_fill(mask == 0, 0.0)
            similarities = similarities.masked_fill(mask == 0, 0.0)

        # Normalize to probability distributions (P and Q)
        q_distribution = torch.clamp(q_matrix / (q_matrix.sum() + eps), min=eps)
        p_distribution = torch.clamp(similarities / (similarities.sum() + eps), min=eps)

        # KL divergence: KL(P || Q)
        kl_div = torch.sum(p_distribution * torch.log(p_distribution / q_distribution))

        return kl_div

    def forward(self, x, mode, return_embedding=False):
        data, data_indices, modality_mask, labels = x
        encoded_modalities = []

        for modality_idx in range(self.dataset.num_omics):
            model_output = self.unimodal_encoders[modality_idx](data[modality_idx])
            encoded_modalities.append(model_output)

        aligned_modalities = align_modalities(encoded_modalities, data_indices, labels)
        modality_embeddings = torch.stack(aligned_modalities, dim=1)
        fused_embeddings, _ = self.attention_layer(modality_embeddings, modality_mask)

        graph_data, similarities, edge_mask = self.dataset.build_graph_data(fused_embeddings, mode)

        if return_embedding:
            output, gnn_embeddings = self.gnn_decoder(graph_data.x, graph_data.edge_index, graph_data.edge_attr, return_embedding=True)
            return aligned_modalities, fused_embeddings, gnn_embeddings, graph_data

        output = self.gnn_decoder(graph_data.x, graph_data.edge_index, graph_data.edge_attr)

        return output, graph_data, similarities, fused_embeddings, edge_mask

    def training_step(self, batch, batch_idx):
        output, graph_data, similarities, fused_embeddings, edge_mask = self(batch, mode="train")
        train_kl_loss = self.similarity_kl_loss(similarities, fused_embeddings, edge_mask)

        mask = graph_data.train_mask
        y_true = graph_data.y.squeeze()[mask]
        y_logit = output[mask]
        train_cls_loss = self.loss_fn(y_logit, y_true)

        total_loss = train_cls_loss + 0.1 * train_kl_loss

        probs = torch.softmax(y_logit, dim=-1)
        metrics = evaluate_classification_performance(y_true, probs, self.dataset.num_classes)

        self.log('train_kl_loss', train_kl_loss.detach().cpu(), prog_bar=True)
        self.log('train_cls_loss', train_cls_loss.detach().cpu(), prog_bar=True)
        self.log('train_total_loss', total_loss.detach().cpu(), prog_bar=True)

        for name, value in metrics.items():
            self.log(f'train_{name}', value, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        if self.dataset.tune_hyperparameters:
            output, graph_data, similarities, fused_embeddings, edge_mask = self(batch, mode="val")
            val_kl_loss = self.similarity_kl_loss(similarities, fused_embeddings, edge_mask)

            mask = graph_data.val_mask
            y_true = graph_data.y.squeeze()[mask]
            y_logit = output[mask]
            val_cls_loss = self.loss_fn(y_logit, y_true)

            total_loss = val_cls_loss + 0.1 * val_kl_loss

            probs = torch.softmax(y_logit, dim=-1)
            metrics = evaluate_classification_performance(y_true, probs, self.dataset.num_classes)

            self.log('val_kl_loss', val_kl_loss.detach().cpu(), prog_bar=True)
            self.log('val_cls_loss', val_cls_loss.detach().cpu(), prog_bar=True)
            self.log('val_total_loss', total_loss.detach().cpu(), prog_bar=True)

            for name, value in metrics.items():
                self.log(f'val_{name}', value, prog_bar=True)

    def test_step(self, batch, batch_idx):
        output, graph_data, similarities, fused_embeddings, edge_mask = self(batch, mode="test")
        test_kl_loss = self.similarity_kl_loss(similarities, fused_embeddings, edge_mask)

        mask = graph_data.test_mask
        y_true = graph_data.y.squeeze()[mask]
        y_logit = output[mask]
        test_cls_loss = self.loss_fn(y_logit, y_true)

        total_loss = test_cls_loss + 0.1 * test_kl_loss

        probs = torch.softmax(y_logit, dim=-1)
        metrics = evaluate_classification_performance(y_true, probs, self.dataset.num_classes)

        self.log('test_kl_loss', test_kl_loss.detach().cpu(), prog_bar=True)
        self.log('test_cls_loss', test_cls_loss.detach().cpu(), prog_bar=True)
        self.log('test_total_loss', total_loss.detach().cpu(), prog_bar=True)

        for name, value in metrics.items():
            self.log(f'test_{name}', value, prog_bar=True)

        # Paired patient mask
        mask_paired = graph_data.test_mask_paired
        y_true_paired = graph_data.y.squeeze()[mask_paired]
        y_logit_paired = output[mask_paired]
        probs_paired = torch.softmax(y_logit_paired, dim=-1)

        metrics_paired = evaluate_classification_performance(y_true_paired, probs_paired, self.dataset.num_classes)
        for name, value in metrics_paired.items():
            self.log(f'test_paired_{name}', value, prog_bar=False)

        # Unpaired patient mask
        mask_unpaired = graph_data.test_mask_unpaired
        y_true_unpaired = graph_data.y.squeeze()[mask_unpaired]
        if len(y_true_unpaired) != 0:
            y_logit_unpaired = output[mask_unpaired]
            probs_unpaired = torch.softmax(y_logit_unpaired, dim=-1)

            metrics_unpaired = evaluate_classification_performance(y_true_unpaired, probs_unpaired, self.dataset.num_classes)
            for name, value in metrics_unpaired.items():
                if value is not None:
                    self.log(f'test_unpaired_{name}', value, prog_bar=False)
                else:
                    self.log(f'test_unpaired_{name}', float('nan'), prog_bar=False)

    def on_test_end(self) -> None:
        batch = self.dataset[0]
        batch = self.transfer_batch_to_device(batch=batch, device=self.device, dataloader_idx=0)
        aligned_modalities, fused_embeddings, gnn_embeddings, graph_data = self(batch, mode="train", return_embedding=True)
        self.train_embeddings_to_plot, self.train_labels_to_plot = self._process_embeddings(aligned_modalities,
                                                                                            fused_embeddings,
                                                                                            gnn_embeddings,
                                                                                            graph_data,
                                                                                            graph_data.train_mask,
                                                                                            mode="train")

        aligned_modalities, fused_embeddings, gnn_embeddings, graph_data = self(batch, mode="test", return_embedding=True)
        self.test_embeddings_to_plot, self.test_labels_to_plot =  self._process_embeddings(aligned_modalities,
                                                                                           fused_embeddings,
                                                                                           gnn_embeddings,
                                                                                           graph_data,
                                                                                           graph_data.test_mask,
                                                                                           mode="test")

    def _process_embeddings(self, aligned_modalities, fused_embeddings, gnn_embeddings, graph_data, mask, mode):
        embeddings_to_plot = {}
        labels_to_plot = {}

        full_labels = graph_data.y[mask].detach().cpu().numpy()

        for idx, data in enumerate(aligned_modalities):
            mask_data = data[mask].detach().cpu().numpy()
            non_zero_rows = ~torch.all(data[mask] == 0, dim=1).detach().cpu().numpy()
            filter_data = mask_data[non_zero_rows]
            filter_labels = full_labels[non_zero_rows]

            if mode == "train":
                embeddings_to_plot[f"Input ({self.dataset.modalities[idx]})"] = self.dataset.get_data(
                    idx).detach().cpu().numpy()[:len(filter_data)]
            else:
                embeddings_to_plot[f"Input ({self.dataset.modalities[idx]})"] = self.dataset.get_data(
                    idx).detach().cpu().numpy()[-len(filter_data):]
            labels_to_plot[f"Input ({self.dataset.modalities[idx]})"] = filter_labels

            embeddings_to_plot[f"Embedded ({self.dataset.modalities[idx]})"] = filter_data
            labels_to_plot[f"Embedded ({self.dataset.modalities[idx]})"] = filter_labels

        fused_mask = fused_embeddings[mask].detach().cpu().numpy()
        gnn_mask = gnn_embeddings[mask].detach().cpu().numpy()

        embeddings_to_plot["Fused embedding"] = fused_mask
        labels_to_plot["Fused embedding"] = full_labels
        embeddings_to_plot["GNN embedding"] = gnn_mask
        labels_to_plot["GNN embedding"] = full_labels

        return embeddings_to_plot, labels_to_plot

    def _custom_data_loader(self):
        return self.dataset

    def train_dataloader(self):
        return self._custom_data_loader()

    def val_dataloader(self):
        return self._custom_data_loader()

    def test_dataloader(self):
        return self._custom_data_loader()