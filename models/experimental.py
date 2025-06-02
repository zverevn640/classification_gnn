import torch
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.nn import MessagePassing
from collections import deque
import numpy as np


# an experimental layer that uses project neighbors features into eigenspace
# then aggregates and projects them back to node feature space
class LocalLaplacianMessagePassing(torch.nn.Module):
    def __init__(self, num_features, dY, sub_nodes):
        super().__init__()
        self.num_features = num_features
        self.sub_nodes = sub_nodes
        self.dY = dY  # dY

    def get_subgraph(self, edge_index, x, node_idx, sub_nodes):
        num_nodes = x.shape[0]

        # Convert edge_index to adjacency list for efficient neighbor lookup
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[src].append(dst)
            adj_list[dst].append(src)
        
        # bfs
        visited = set([node_idx])
        queue = deque([node_idx])
        
        while len(queue) > 0 and len(visited) < sub_nodes:
            current = queue.popleft()
            for neighbor in adj_list[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if len(visited) >= sub_nodes:
                        break
        
        subgraph_nodes = sorted(list(visited))
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph_nodes)}
        
        # extract edges
        num_edges = edge_index.size(1)
        mask = torch.zeros(num_edges, dtype=torch.bool)
        for i in range(num_edges):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in visited and dst in visited:
                mask[i] = True

        subgraph_edge_index = edge_index[:, mask].clone()

        # remap nodes
        for i in range(subgraph_edge_index.size(1)):
            subgraph_edge_index[0, i] = node_map[subgraph_edge_index[0, i].item()]
            subgraph_edge_index[1, i] = node_map[subgraph_edge_index[1, i].item()]
        
        subgraph_x = x[subgraph_nodes]

        return {
            "x": subgraph_x,
            "edge_index": subgraph_edge_index,
            "original_idx": torch.tensor(subgraph_nodes),
            "node_map": node_map,
            "num_nodes": len(subgraph_nodes)
        }

    
    def compute_laplacian_eigenfuncs(self, subgraph):
        """Compute first dY eigenfunctions of the graph Laplacian for the subgraph"""
        adj_matrix = to_scipy_sparse_matrix(subgraph["edge_index"], 
                                           num_nodes=subgraph["num_nodes"])
        
        # Compute normalized Laplacian of the subgraph
        degree = np.array(adj_matrix.sum(1)).flatten()
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # L = I - D^(-1/2) * A * D^(-1/2)
        laplacian = sp.eye(subgraph["num_nodes"]) - d_mat_inv_sqrt.dot(adj_matrix).dot(d_mat_inv_sqrt)

        k = min(self.dY, subgraph["num_nodes"] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
        # eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]

        return torch.from_numpy(eigenvectors).float()
    
    def get_neighbors(self, edge_index, node_idx):
        """Get neighbors of vertex node_idx from edge_index"""
        neighbors = set()
        for i in range(edge_index.size(1)):
            src = edge_index[0][i].item()
            dst = edge_index[1][i].item()

            if src == node_idx:
                neighbors.add(dst)
            elif dst == node_idx:
                neighbors.add(src)

        return list(neighbors)
    
    def project_features(self, subgraph, eigenfuncs):
        x = subgraph["x"]
        return (eigenfuncs.t() @ x) * subgraph["num_nodes"]
    
    def reconstruct_features(self, eigenfuncs, c):
        return eigenfuncs @ c

    def forward(self, x, edge_index, k_hop=1):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        num_nodes = x.size(0)
        x_new = torch.zeros_like(x)
        neighbors_counts = {}

        for v in range(num_nodes):
            subgraph_v = self.get_subgraph(edge_index, x, v, self.sub_nodes)

            eigenfuncs_v = self.compute_laplacian_eigenfuncs(subgraph_v)

            neighbors = self.get_neighbors(edge_index, v)
            neighbors_counts[v] = len(neighbors)
            for u in neighbors:
                subgraph_u = self.get_subgraph(edge_index, x, u, self.sub_nodes)

                eigenfuncs_u = self.compute_laplacian_eigenfuncs(subgraph_u)

                c_u = self.project_features(subgraph_u, eigenfuncs_u)
                
                x_u_at_v = self.reconstruct_features(eigenfuncs_v, c_u).sum(dim=0)
        
                x_new[v] += x_u_at_v

            if len(neighbors) == 0:
                x_new[v] = x[v]
                continue
            # take average
            x_new[v] /= neighbors_counts[v]

        return x_new
    

# import torch
# import scipy.sparse as sp
# import scipy.sparse.linalg as sla
# from torch_geometric.utils import to_scipy_sparse_matrix
# from torch_geometric.nn import MessagePassing
# import numpy as np

# class LaplacianMessagePassing(nn.Module):
#     def __init__(self, num_features, dY):
#         super().__init__(aggr="mean")
#         self.num_features = num_features
#         self.dY = dY  # dY

#     def get_subgraph(self, edge_index, x, node_idx, k_hop):
#         """Extract k-hop subgraph around node_idx"""
#         num_nodes = x.size(0)

#         # bfs
#         visited = set([node_idx])
#         current_level = set([node_idx])

#         for _ in range(k_hop):
#             next_level = set()
#             for node in current_level:
#                 neighbors = edge_index[1, edge_index[0] == node].tolist()
#                 # neighbors += edge_index[0, edge_index[1] == node].tolist()
#                 next_level.update(neighbors)

#             visited.update(next_level)
#             current_level = next_level

#         subgraph_nodes = sorted(list(visited))
#         node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph_nodes)}

#         # extract edges
#         num_edges = edge_index.size(1)
#         mask = torch.zeros(num_edges, dtype=torch.bool)
#         for i in range(num_edges):
#             src, dst = edge_index[0, i].item(), edge_index[1, i].item()
#             if src in visited and dst in visited:
#                 mask[i] = True

#         subgraph_edge_index = edge_index[:, mask].clone()

#         # remap nodes
#         for i in range(subgraph_edge_index.size(1)):
#             subgraph_edge_index[0, i] = node_map[subgraph_edge_index[0, i].item()]
#             subgraph_edge_index[1, i] = node_map[subgraph_edge_index[1, i].item()]
        
#         subgraph_x = x[subgraph_nodes]

#         return {
#             "x": subgraph_x,
#             "edge_index": subgraph_edge_index,
#             "original_idx": torch.tensor(subgraph_nodes),
#             "node_map": node_map,
#             "num_nodes": len(subgraph_nodes)
#         }
    
#     def compute_laplacian_eigenfuncs(self, subgraph):
#         """Compute first dY eigenfunctions of the graph Laplacian for the subgraph"""
#         adj_matrix = to_scipy_sparse_matrix(subgraph["edge_index"], 
#                                            num_nodes=subgraph["num_nodes"])
        
#         # Compute normalized Laplacian of the subgraph
#         degree = np.array(adj_matrix.sum(1)).flatten()
#         d_inv_sqrt = np.power(degree, -0.5)
#         d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#         d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

#         # L = I - D^(-1/2) * A * D^(-1/2)
#         laplacian = sp.eye(subgraph["num_nodes"]) - d_mat_inv_sqrt.dot(adj_matrix).dot(d_mat_inv_sqrt)

#         k = min(self.dY, subgraph["num_nodes"] - 1)
#         eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
#         # eigenvalues = eigenvalues[:k]
#         eigenvectors = eigenvectors[:, :k]

#         return torch.from_numpy(eigenvectors).float()
    
#     def get_neighbors(self, edge_index, node_idx):
#         """Get neighbors of vertex node_idx from edge_index"""
#         neighbors = set()
#         for i in range(edge_index.size(1)):
#             src = edge_index[0][i].item()
#             dst = edge_index[1][i].item()

#             if src == node_idx:
#                 neighbors.add(dst)
#             elif dst == node_idx:
#                 neighbors.add(src)

#         return list(neighbors)
    
#     def project_features(self, subgraph, eigenfuncs):
#         x = subgraph["x"]
#         return (eigenfuncs.t() @ x) * subgraph["num_nodes"]
    
#     def reconstruct_features(self, eigenfuncs, c):
#         return eigenfuncs @ c

#     def forward(self, edge_index, x, k_hop=1):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]
#         num_nodes = x.size(0)
#         x_new = torch.zeros_like(x)
#         neighbors_counts = {}

#         for v in range(num_nodes):
#             subgraph_v = self.get_subgraph(edge_index, x, v, k_hop)

#             eigenfuncs_v = self.compute_laplacian_eigenfuncs(subgraph_v)

#             neighbors = self.get_neighbors(edge_index, v)
#             neighbors_counts[v] = len(neighbors)
#             for u in neighbors:
#                 subgraph_u = self.get_subgraph(edge_index, x, u, k_hop)

#                 eigenfuncs_u = self.compute_laplacian_eigenfuncs(subgraph_u)

#                 c_u = self.project_features(subgraph_u, eigenfuncs_u)

#                 x_u_at_v = self.reconstruct_features(eigenfuncs_v, c_u).sum(dim=0)
        
#                 x_new[v] += x_u_at_v

#             if len(neighbors) == 0:
#                 x_new[v] = x[v]
#                 continue
#             # take average
#             x_new[v] /= neighbors_counts[v]

#         return x_new