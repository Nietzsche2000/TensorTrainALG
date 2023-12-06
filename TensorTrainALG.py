import numpy as np
import tensornetwork as tn


def rank_reduced_svd(M, r):
    """
    Perform a rank-reduced singular value decomposition (SVD) on a matrix.

    Args:
    M (np.ndarray): The matrix to be decomposed.
    r (int): The number of singular values and vectors to retain.

    Returns:
    tuple: The truncated U, S, and Vh matrices of the SVD.
    """
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    U_reduced = U[:, :r]
    S_reduced = np.diag(S[:r])
    Vh_reduced = Vh[:r, :]
    return U_reduced, S_reduced, Vh_reduced


def tt(node):
    """
    Create a tensor train (TT) decomposition of a given tensor.

    Args:
    node (tn.Node): A node representing a tensor in the tensor network.

    Returns:
    list: A list of tensor cores forming the tensor train.
    """
    tensor = node.tensor.copy()
    C = tensor
    cores = []
    r_k_1 = 1
    for k in range(1, len(tensor.shape)):
        v = r_k_1 * tensor.shape[k]
        C = np.reshape(C, (v, C.size // v))
        rk = np.linalg.matrix_rank(C)
        U, S, Vh = rank_reduced_svd(C, rk)
        G = np.reshape(U, (r_k_1, tensor.shape[k], rk))
        cores.append(tn.Node(G))
        C = S @ Vh
        r_k_1 = rk
    cores.append(tn.Node(np.reshape(C, (C.shape[0], C.shape[1], 1))))
    return cores


def tt_rounding(cores, r):
    """
    Apply rounding to a tensor train to reduce its rank.

    Args:
    cores (list): List of tensor cores in the tensor train.
    r (list): Desired rank of each core after rounding.

    Returns:
    list: List of rounded tensor cores.
    """
    G = cores.copy()
    for k in range(len(G) - 1, 0, -1):
        # BACKWARD PASS
        Gk = G[k]
        Gk_shape = Gk.tensor.shape
        Gk_mat = Gk.tensor.reshape(Gk_shape[0], Gk_shape[1] * Gk_shape[2])
        Q, R = np.linalg.qr(Gk_mat.T)
        Gk_new = Q.T.reshape(Gk_shape)
        G[k] = tn.Node(Gk_new)
        G_k_1 = G[k - 1]
        ed = G_k_1[2] ^ tn.Node(R)[0]
        G[k - 1] = tn.contract(ed)

    # FORWARD PASS
    for k in range(0, len(G) - 1):
        Gk = G[k]
        Gk_shape = Gk.tensor.shape
        Gk_mat = Gk.tensor.reshape(Gk_shape[0] * Gk_shape[1], Gk_shape[2])
        Gk_trunc_mat, sigma, VT = rank_reduced_svd(Gk_mat, r[k])
        G[k] = tn.Node(Gk_trunc_mat.reshape((Gk_shape[0], Gk_shape[1], r[k])))
        ed = tn.Node(sigma @ VT)[1] ^ G[k + 1][0]
        G[k + 1] = tn.contract(ed)
    return G


def attach(cores):
    """
    Connect the cores of a tensor train.

    Args:
    cores (list): List of tensor cores in the tensor train.

    Returns:
    list: List of connected edges in the tensor train.
    """
    connected_edges = []
    for i in range(1, len(cores)):
        core_left = cores[i - 1]
        core_right = cores[i]
        ed = core_left[len(core_left.get_all_edges()) - 1] ^ core_right[0]
        connected_edges.append(ed)
    return connected_edges


def tt_to_tensor(tt_conn_edges):
    """
    Convert a tensor train back into a regular tensor.

    Args:
    tt_conn_edges (list): List of connected edges in the tensor train.

    Returns:
    tn.Node: A node representing the reconstructed tensor.
    """
    M = []
    for edge in tt_conn_edges:
        M = tn.contract(edge)
    return tn.Node(M.tensor.reshape((M.tensor.shape[1:][:-1])))


def t_norm(node):
    """
    Calculate the Frobenius norm of a tensor represented by a TensorNetwork node.

    The Frobenius norm, often used in tensor computations, is the square root of the sum of the squares of all elements in the tensor.

    Args:
    node (tn.Node): A node representing a tensor in the tensor network.

    Returns:
    float: The Frobenius norm of the tensor.
    """
    squared_sum = np.sum(node.tensor * node.tensor)
    return np.sqrt(squared_sum)
