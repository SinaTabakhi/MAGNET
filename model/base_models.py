import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor


class MLPEncoder(nn.Module):
    def __init__(self, in_dims, hid_dims, dropout_rate: float = 0.0, negative_slope: float = 0.2):
        super().__init__()

        self.encoder_layers = nn.Sequential(
            Linear(in_dims, hid_dims, weight_initializer='glorot', bias_initializer='zeros'),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(p=dropout_rate),
            Linear(hid_dims, hid_dims, weight_initializer='glorot', bias_initializer='zeros')
        )

    def forward(self, x):
        x = self.encoder_layers(x)
        return x


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dims, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dims = hid_dims // num_heads
        assert (
            self.head_dims * num_heads == hid_dims
        ), "hid_dims must be divisible by num_heads"

        self.lin_proj = Linear(hid_dims, hid_dims, bias=False, weight_initializer='glorot')
        self.att_lin = nn.Parameter(torch.empty(num_heads, 1, self.head_dims))
        self.out_proj = Linear(hid_dims, hid_dims, bias=False, weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_proj.reset_parameters()
        self.out_proj.reset_parameters()
        glorot(self.att_lin)

    def forward(self, x, mask):
        """
        x: [batch_size, num_modalities, hid_dims] - Patient embeddings for all modalities
        mask: [batch_size, num_modalities] - Mask indicating available modalities
        """
        batch_size, num_modalities, hid_dims = x.size()

        # Linear projection
        x_proj = self.lin_proj(x).view(batch_size, num_modalities, self.num_heads, self.head_dims)  # [batch_size, num_modalities, num_heads, head_dims]
        x_proj = x_proj.permute(0, 2, 1, 3) # [batch_size, num_heads, num_modalities, head_dims]

        # Compute attention scores
        att_scores = torch.matmul(x_proj, self.att_lin.transpose(-1, -2)).squeeze(-1)  # [batch_size, num_heads, num_modalities]
        att_scores = att_scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))  # [batch_size, num_heads, num_modalities]

        # Compute attention weights
        att_weights = torch.softmax(att_scores, dim=-1) # [batch_size, num_heads, num_modalities]
        att_weights = att_weights * mask.unsqueeze(1) # [batch_size, num_heads, num_modalities]
        fused_embeddings = torch.sum(att_weights.unsqueeze(-1) * x_proj, dim=2) # [batch_size, num_heads, head_dim]

        # Concatenate heads and project output
        fused_embeddings = fused_embeddings.view(batch_size, -1)  # [batch_size, hid_dims]
        output = self.out_proj(fused_embeddings) # [batch_size, hid_dims]

        return output, att_weights


class EdgeSAGEConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr = "mean",
        bias: bool = True,
        edge_dim: int = None,
        **kwargs,
    ):
        super().__init__(aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        in_channels = (in_channels, in_channels)

        if self.edge_dim is not None:
            self.lin_msg = Linear(in_channels[0] + self.edge_dim, in_channels[0], weight_initializer='glorot', bias_initializer='zeros', bias=True)

        self.lin = Linear(in_channels[0], in_channels[0], weight_initializer='glorot', bias_initializer='zeros', bias=True) # Linear projection of source node features
        self.lin_l = Linear(in_channels[0], out_channels, weight_initializer='glorot', bias_initializer='zeros', bias=bias) # Linear layer on aggregated neighbor info
        self.lin_r = Linear(in_channels[1], out_channels, weight_initializer='glorot', bias_initializer='zeros', bias=False) # Linear layer on root (target) node

        self.act_msg = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.edge_dim is not None:
            self.lin_msg.reset_parameters()

    def forward(self, x, edge_index, edge_attr = None):
        if isinstance(x, Tensor):
            x = (x, x)

        x = (self.lin(x[0]).relu(), x[1])

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.lin_l(out)
        x_r = x[1]
        out = out + self.lin_r(x_r)

        return out

    def message(self, x_j, edge_attr = None):
        if edge_attr is not None and self.edge_dim is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)  # Ensure 2D shape
            msg = torch.cat([x_j, edge_attr], dim=-1)
            return self.act_msg(self.lin_msg(msg))
        return x_j


class GNNDecoder(nn.Module):
    def __init__(self, hid_dims, out_dims, num_layers, dropout_rate: float = 0.0, negative_slope: float = 0.2):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.negative_slope = negative_slope
        self.conv_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.conv_layers.append(EdgeSAGEConv(hid_dims, hid_dims, edge_dim=1))

        self.decoder_layers = nn.Sequential(
            Linear(hid_dims, hid_dims, weight_initializer='glorot', bias_initializer='zeros'),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.final_layer = Linear(hid_dims, out_dims, weight_initializer='glorot', bias_initializer='zeros')


    def forward(self, x, edge_index, edge_attr=None, return_embedding=False):
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.decoder_layers(x)
        embeddings = x

        logits = self.final_layer(x)

        if return_embedding:
            return logits, embeddings

        return logits