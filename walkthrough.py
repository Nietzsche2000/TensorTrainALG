import numpy as np

from TensorTrainALG import *


def test_tt_decomp():
    # GENERATE A RANDOM MATRIX
    M = np.random.rand(2, 2, 2)

    # MAKE THE TENSOR NODE--ONE BIG TENSOR
    node = tn.Node(M)

    # MAKE THE TENSOR TRAIN FROM THE TENSOR
    cores = tt(node)

    # ATTACH ALL THE CORES BACK
    connected_edges = attach(cores)

    # TT TO TENSOR
    reconstructed_node = tt_to_tensor(connected_edges)

    # VERIFY WHETHER IT WORKS BY COMPUTING THE NORM OF THE DIFFERENCE BETWEEN M2 AND M
    print("THE ERROR FOR TT TO TENSOR: ", t_norm(node - reconstructed_node))


def test_tt_rounding():
    # NOTE GENERALLY RANDOM MATRICES ARE INCOMPRESSIBLE
    # THE ERROR FOR TT COMPRESSION: 2.9683464371643897e-14 WITH RANK: (10, 10, 10)
    # GENERATE A RANDOM MATRIX
    M = np.random.rand(10, 10, 10)

    for rank in range(1, 11):
        # MAKE THE TENSOR NODE--ONE BIG TENSOR
        node = tn.Node(M)

        # MAKE THE TENSOR TRAIN FROM THE TENSOR
        cores = tt(node)

        # COMPRESS THE CORES
        cores_reduced = tt_rounding(cores, [rank, rank, rank])

        # ATTACH ALL THE CORES BACK
        connected_edges = attach(cores)
        connected_edges_reduced = attach(cores_reduced)

        # TT TO TENSOR
        reconstructed_node = tt_to_tensor(connected_edges)
        reconstructed_node_compressed = tt_to_tensor(connected_edges_reduced)

        # VERIFY WHETHER IT WORKS BY COMPUTING THE NORM OF THE DIFFERENCE BETWEEN M2 AND M
        print(
            f"THE ERROR FOR TT COMPRESSION: {t_norm(reconstructed_node - reconstructed_node_compressed)} WITH RANK: {(rank, rank, rank)}")
