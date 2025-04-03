import numpy as np
from tqdm import tqdm

########################################
### MOVIE LENS SPECIFIC START        ###
########################################

def mlens_get_subgraph(mat, users_num, max_depth=1, max_edges=10):
    """get an adjacency list of a subgraph from MovieLens dataset

    Args:
        mat: matrix of shape (num_users, num_movies) were a_ij is the rating of i-th user for j-th movie
        max_depth (int, optional): Defaults to 1.
        max_edges (int, optional): Defaults to 10. How many neighbors of a node get grabbed during processing

    Returns:
        list[tuple(int,int)]: adjacency list [(user_id, movie_id)] representing a sampled subgraph
    """
    adjacency_list = set()
    depth = 0

    user = np.random.choice(np.arange(1, users_num + 1))

    new_adj, movies = sample(mat, user, is_user=True, max_edges=max_edges)
    adjacency_list.update(new_adj)
    depth += 1


    while depth < max_depth:
        is_user = False
        users = set()
        for movie_id in movies:
            new_adj, new_users = sample(mat, movie_id, is_user, max_edges)
            adjacency_list.update(new_adj)
            users.update(new_users)
        
        is_user = True
        movies = set()
        for user_id in users:
            new_adj, new_movies = sample(mat, movie_id, is_user, max_edges)
            adjacency_list.update(new_adj)
            movies.update(new_movies)

        depth += 1

    return list(adjacency_list)

def mlens_sample(mat, vertice_id, is_user, max_edges):
    """helper function that is called from get_subgraph. takes random 'max_edges' number of edges"""
    adjacency_list = []

    # get ids of all vertices connected to vertice_id
    if is_user:
        vert_neigh = np.where(mat[vertice_id - 1] > 0)[0] # movies that were rated by user
    else:
        vert_neigh = np.where(mat[:, vertice_id - 1] > 0)[0] # users that rated the movie

    num_edges = min(max_edges, len(vert_neigh))

    neighs_ids = np.random.choice(vert_neigh, num_edges, replace=False)
    neighs_ids = [idx + 1 for idx in neighs_ids] # shift values to match ids
    
    # add edges that represent user ratings: tuple(user, movie)
    for vert in neighs_ids:
        edge = (vertice_id, vert) if is_user else (vert, vertice_id)
        adjacency_list.append(edge)

    return adjacency_list, neighs_ids


def mlens_get_adj_matrix(adj_list, mat):
    """given adjacency list adj_list and rating mat, outputs adjacency matrix A with user and movie mappings"""

    # mapping that represents user_id -> adjacency_mat_id; and same for movies
    users_map = dict()
    movies_map = dict()

    next_idx = 0

    for user, movie in adj_list:
        if user not in users_map:
            users_map[user] = next_idx
            next_idx += 1
        if movie not in movies_map:
            movies_map[movie] = next_idx
            next_idx += 1

    n = len(users_map.keys()) + len(movies_map.keys())
    A = np.zeros((n, n))

    for user, movie in adj_list:
        u_id = users_map[user]
        m_id = movies_map[movie]
    
        # since original user_id (movie_id) starts from 1
        rating = mat[user - 1][movie - 1]

        A[u_id][m_id] = rating
        A[m_id][u_id] = rating

    return A, users_map, movies_map

########################################
### MOVIE LENS SPECIFIC END          ###
########################################


def get_eigenmap_from_edge_index(edge_index, num_nodes, d=2):
    """Find eigenmap of a laplacian associated with adjacency matrix A, d - number of eigenvecs"""

    A = get_adjacency(edge_index, num_nodes)

    D = np.zeros_like(A)
    np.fill_diagonal(D, np.sum(A,axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    I = np.eye(A.shape[0])
    L = I - np.dot(D_inv_sqrt, A).dot(D_inv_sqrt)

    return get_eigenmap(L, d) 

def get_eigenmap(L, d=2):
    """Find laplacian L eigenmap composed from d eigenvectors"""

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the eigenvectors corresponding to the smallest eigenvalues
    eigenmaps = eigenvectors[:, 1:d+1]

    return eigenmaps

def get_adjacency(edge_index, num_nodes):
    """Get adjacency matrix from edge list, edge_list shape = (2, x), x - number of edges"""

    A = np.zeros((num_nodes, num_nodes))
    num_edges = edge_index.shape[1]

    for i in range(num_edges):
        src = edge_index[0][i]
        dst = edge_index[1][i]

        A[src][dst] = 1
        A[dst][src] = 1

    return A

# def local_pca_alignment(X, edge_index, stalk_dim):
    
#     num_nodes = X.shape[0]
#     idx_to_o = dict()

#     for i in tqdm(range(num_nodes)):
#         X_cap = local_neighbourhood(X, edge_index, i, stalk_dim)
#         X_cap = X_cap.T

#         U, _, _ = np.linalg.svd(X_cap)
#         O_i = U[:, :stalk_dim]
#         idx_to_o[i] = O_i

#     processed_edges = set()
#     conn_lap = np.zeros((num_nodes*stalk_dim, num_nodes*stalk_dim))

#     num_edges = edge_index.shape[1]
#     for edge_id in tqdm(range(num_edges)):
#         src = edge_index[0][edge_id]
#         dst = edge_index[1][edge_id]

#         node_pair = tuple(sorted(src,dst))
        
#         if node_pair in processed_edges or src == dst:
#             continue

#         O_i = idx_to_o[src]
#         O_j = idx_to_o[dst]
#         U, _, Vh = np.linalg.svd(O_i.T @ O_j)
#         O_ij = U*Vh

#         conn_lap[src : src+stalk_dim, dst : dst+stalk_dim] += O_ij
#         conn_lap[src : src+stalk_dim, dst : dst+stalk_dim] += O_ij

#         processed_edges.add(node_pair)

#     return conn_lap

# TODO: bad-bad-bad RAM usage, remake later
def local_pca_alignment(X, edge_index, stalk_dim):

    num_nodes = X.shape[0]
    num_edges = edge_index.shape[1]

    processed_edges = set()
    node_to_o = dict()
    conn_lap = np.zeros((num_nodes*stalk_dim, num_nodes*stalk_dim))

    for edge_id in tqdm(range(num_edges)):
        src = edge_index[0][edge_id]
        dst = edge_index[1][edge_id]

        if src == dst:
            continue

        node_pair = tuple(sorted((src, dst)))
        if node_pair in processed_edges:
            continue
        processed_edges.add(node_pair)

        for node in node_pair:
            if node in node_to_o:
                continue
            X_cap = local_neighbourhood(X, edge_index, node, stalk_dim).T

            U, _, _ = np.linalg.svd(X_cap)
            O_i = U[:, :stalk_dim]
            node_to_o[node] = O_i

        O_i = node_to_o[src]
        O_j = node_to_o[dst]
        U, _, Vh = np.linalg.svd(O_i.T @ O_j)
        O_ij = U*Vh

        conn_lap[src : src+stalk_dim, dst : dst+stalk_dim] += O_ij
        conn_lap[src : src+stalk_dim, dst : dst+stalk_dim] += O_ij

        del node_to_o[src] # TODO: change that
        del node_to_o[dst] 

    return conn_lap

def local_neighbourhood(X, edge_index, i, stalk_dim):
    # pick d neighbors in 1-hop region, if less than d 1-hop neihbors exist, pick based on eucl. dist
    if X.shape[0] < stalk_dim:
        raise ValueError("nodes num in feature matrix X is less than stalk_dim")

    num_edges = edge_index.shape[1]

    processed_neighs = set()
    processed_neighs.add(i)

    X_cap = []

    for edge_id in range(num_edges):
        src = edge_index[0][edge_id]
        dst = edge_index[1][edge_id]
    
        neigh = src if src != i else dst

        if i in (src, dst) and neigh not in processed_neighs:
            processed_neighs.add(neigh)
            X_cap.append(X[neigh] - X[i])

    # X_cap shape = num_neighs x feature_dim
    num_neighs = len(X_cap)#X_cap.shape[0]

    if num_neighs >= stalk_dim:
        return np.array(X_cap)
    

    # not enough neighbors, pick more based on euclidean distance
    needed = stalk_dim - num_neighs

    diffs = X - X[i]
    dists = np.sqrt(np.sum(diffs**2, axis=1))

    for node_id in np.argsort(dists):
        if node_id in processed_neighs:
            continue

        needed -= 1
        X_cap.append(X[node_id] - X[i])

        if needed <= 0:
            break

    return np.array(X_cap)
