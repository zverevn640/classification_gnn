import torch
import torch.nn as nn
import torch.nn.functional as F

class SheafDiffusion(nn.Module):
    def __init__(self, edge_index, args, conn_lap):
        super().__init__()
        
        self.input_dim = args["input_dim"]
        self.hidden_channels = args["hidden_channels"]
        self.out_dim = args["output_dim"]
        self.graph_sz = args["graph_size"]
        self.d = args["d"]
        self.hidden_dim = self.hidden_channels * self.d

        self.num_layers = args["num_layers"]
        self.dropout = 0.7276458263736642
        self.edge_index = edge_index
        self.conn_lap = conn_lap

        self.left_w = nn.ModuleList()
        self.right_w = nn.ModuleList()
        self.eps = nn.ParameterList()
        for i in range(self.num_layers):
            self.right_w.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
            nn.init.orthogonal_(self.right_w[-1].weight.data)

            self.left_w.append(nn.Linear(self.d, self.d, bias=False))
            nn.init.eye_(self.left_w[-1].weight.data)

            self.eps.append(nn.Parameter(torch.zeros((self.d, 1))))


        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.out_dim)
    
    def forward(self, x):
        x = self.lin1(x)

        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = x.view(self.graph_sz * self.d, -1)

        x0 = x
        for layer in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x.t().reshape(-1, self.d)
            x = self.left_w[layer](x)
            x = x.reshape(-1, self.graph_sz * self.d).t()

            x = self.right_w[layer](x)

            x = self.conn_lap @ x
            x = F.elu(x)
            x0 = (1 + torch.tanh(self.eps[layer]).tile(self.graph_sz, 1)) * x0 - x
            x = x0

        x = x.reshape(self.graph_sz, -1)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)