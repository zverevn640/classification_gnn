import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

class ContenderModel(torch.nn.Module):
    def __init__(self, edge_index, args):
        super().__init__()

        self.input_size = args["input_dim"]
        self.output_size = args["output_dim"]
        self.hidden_size = args["hidden_dim"]

        self.edge_index = edge_index

        if args["model"] == "GCN":
            self.conv1 = GCNConv(self.input_size, self.hidden_size)
            self.conv2 = GCNConv(self.hidden_size, self.output_size)
        elif args["model"] == "GAT":
            self.conv1 = GATConv(self.input_size, self.hidden_size)
            self.conv2 = GATConv(self.hidden_size, self.output_size)
        else:
            self.conv1 = SAGEConv(self.input_size, self.hidden_size)
            self.conv2 = SAGEConv(self.hidden_size, self.output_size)

    def forward(self, x):
        x = self.conv1(x, self.edge_index)
        x = x.relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, self.edge_index)
        return F.log_softmax(x, dim=1)