import numpy as np


########################################
### MOVIE LENS SPECIFIC START        ###
########################################

def get_subgraph(mat, users_num, max_depth=1, max_edges=10):
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

def sample(mat, vertice_id, is_user, max_edges):
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


def get_adj_matrix(adj_list, mat):
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


def get_eigenmap(A):
    """Find eigenmap of a laplacian associated with adjacency matrix A"""

    D = np.zeros_like(A)
    np.fill_diagonal(D, np.sum(A,axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    I = np.eye(A.shape[0])
    L = I - np.dot(D_inv_sqrt, A).dot(D_inv_sqrt)

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the eigenvectors corresponding to the smallest eigenvalues
    d = 2
    eigenmaps = eigenvectors[:, 1:d+1]

    return eigenmaps