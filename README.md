# TensorTrainALG
## About
This project was an outcome of the UC Berkeley Mathematics Directed Reading Program. This project was built under the guidance of PhD student Michael Kielstra.

## Description
`TensorTrainALG.py` is a Python script focused on advanced tensor operations, leveraging tensor train algorithms and singular value decomposition (SVD). It is designed for use in areas like machine learning, data analysis, and scientific computing and depends on the `numpy` and `tensornetwork` libraries.

## Installation
Ensure Python is installed, along with the necessary libraries. Install `numpy` and `tensornetwork` using pip:

```bash
pip install numpy tensornetwork
```

## Usage
The script is designed to be used as a module or incorporated into larger projects. An example of how to use the script's functionalities can be found in `walkthrough.py`.

## Functions

### `rank_reduced_svd(M, r)`
Performs rank-reduced singular value decomposition on a matrix.
- **Args:**
  - `M (np.ndarray)`: The matrix to be decomposed.
  - `r (int)`: Number of singular values and vectors to retain.
- **Returns:**
  - Tuple of truncated U, S, and Vh matrices of the SVD.

### `tt(node)`
Creates a tensor train (TT) decomposition of a tensor.
- **Args:**
  - `node (tn.Node)`: A node representing a tensor in the tensor network.
- **Returns:**
  - List of tensor cores forming the tensor train.

### `tt_rounding(cores, r)`
Applies rounding to a tensor train to reduce its rank.
- **Args:**
  - `cores (list)`: List of tensor cores in the tensor train.
  - `r (list)`: Desired rank of each core after rounding.
- **Returns:**
  - List of rounded tensor cores.

### `attach(cores)`
Connects the cores of a tensor train.
- **Args:**
  - `cores (list)`: List of tensor cores in the tensor train.
- **Returns:**
  - List of connected edges in the tensor train.

### `tt_to_tensor(tt_conn_edges)`
Converts a tensor train back into a regular tensor.
- **Args:**
  - `tt_conn_edges (list)`: List of connected edges in the tensor train.
- **Returns:**
  - A node representing the reconstructed tensor.

### `t_norm(node)`
Calculates the Frobenius norm of a tensor represented by a TensorNetwork node.
- **Args:**
  - `node (tn.Node)`: A node representing a tensor in the tensor network.
- **Returns:**
  - The Frobenius norm of the tensor.

## Contributing
Contributions to improve or enhance the script are welcome. Fork the repository, make your changes, and submit a pull request.

## License
This script is released under the [MIT License](https://opensource.org/licenses/MIT).

## Contact
For any queries or support, please contact monishwaran@berkeley.edu.
