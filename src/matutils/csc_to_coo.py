import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def load_sym_csc(filename):
    with open(filename, "r") as file:
        # Read the number of rows/columns and number of non-zero elements
        n = int(file.readline().strip())
        n = int(file.readline().strip())  # Assuming the file repeats the dimension
        nnz = int(file.readline().strip())

        # Allocate memory for CSC arrays
        ia = np.zeros(n + 1, dtype=np.int32)
        ja = np.zeros(nnz, dtype=np.int32)
        a = np.zeros(nnz, dtype=np.float64)

        # Read the CSC arrays from the file
        for i in range(nnz):
            ja[i] = int(file.readline().strip())

        for i in range(n + 1):
            ia[i] = int(file.readline().strip())

        for i in range(nnz):
            a[i] = float(file.readline().strip())

    return n, nnz, ia, ja, a


def save_coo_bin(A: sp.coo_matrix, filename: str):
    n = A.shape[0]
    nnz = A.nnz
    row = A.row.astype(np.int32)
    col = A.col.astype(np.int32)
    data = A.data.astype(np.float64)

    with open(filename, 'wb') as f:
        f.write(n.to_bytes(4, byteorder='little'))
        f.write(n.to_bytes(4, byteorder='little'))
        f.write(nnz.to_bytes(4, byteorder='little'))
        f.write(row.tobytes())
        f.write(col.tobytes())
        f.write(data.tobytes())


if __name__ == "__main__":
    n, nnz, ia, ja, a = load_sym_csc("/capstor/scratch/cscs/vmaillou/data/bta_dataset/Qxy_ns2865_nt365_nss0_nb4_n1045729.dat")
    # n, nnz, ia, ja, a = load_sym_csc("/capstor/scratch/cscs/vmaillou/data/bta_dataset/Qxy_ns42_nt3_nss0_nb2_n128.dat")
    
    A_csc = sp.csc_matrix((a, ja, ia), shape=(n, n))
    A_coo = A_csc.tocoo()

    """ # Filter out the entries above the diagonal
    mask = A_coo.row <= A_coo.col
    row = A_coo.row[mask]
    col = A_coo.col[mask]
    data = A_coo.data[mask]

    # Create the symmetric COO matrix
    A_coo_upper = sp.coo_matrix((data, (row, col)), shape=(n, n)) """

    # Save the COO matrix to a binary file
    save_coo_bin(A_coo, "/capstor/scratch/cscs/vmaillou/data/bta_dataset/Qxy_ns2865_nt365_nss0_nb4_n1045729_mumps.dat")
    # save_coo_bin(A_coo, "/capstor/scratch/cscs/vmaillou/data/bta_dataset/Qxy_ns42_nt3_nss0_nb2_n128_mumps.dat")


