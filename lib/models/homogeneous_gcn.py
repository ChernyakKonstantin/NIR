import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn.models import MLP, GCN
from torch_geometric.utils import scatter


class HomogeneousGCN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            gnn_hidden_channels: int = 32,
            gnn_num_layers: int = 3,
            mlp_hidden_channels: int = 32,
            mlp_num_layers: int = 3,
    ):
        super().__init__()
        self.encoder = GCN(
            in_channels=in_channels,
            hidden_channels=gnn_hidden_channels,
            num_layers=gnn_num_layers,
            norm="batch_norm"
        )
        self.mlp = MLP(
            in_channels=self.encoder.out_channels,
            hidden_channels=mlp_hidden_channels,
            out_channels=out_channels,
            num_layers=mlp_num_layers,
            act="relu",
            norm="batch_norm",
            plain_last=True,
        )

    def forward(self, data: Data) -> torch.Tensor:
        z = self.encoder(data.x, data.edge_index)
        z = scatter(z, data.batch, dim=0, reduce='mean') # TODO: Maybe use aggregation layer insted
        return self.mlp(z)
