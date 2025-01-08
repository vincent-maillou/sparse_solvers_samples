import numpy as np
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


def plot_spy(n, nnz, ia, ja, a, output_filename):
    row_indices = []
    col_indices = []
    for col in range(n):
        for idx in range(ia[col], ia[col + 1]):
            row_indices.append(ja[idx])
            col_indices.append(col)

    plt.figure(figsize=(10, 10))
    plt.scatter(col_indices, row_indices, marker='.', color='black')
    plt.gca().invert_yaxis()
    plt.title("Spy Plot of Sparse Matrix")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.savefig(output_filename)
    plt.close()

if __name__ == "__main__":
    n, nnz, ia, ja, a = load_sym_csc("/capstor/scratch/cscs/vmaillou/data/bta_dataset/Qxy_ns2865_nt365_nss0_nb4_n1045729.dat")
    print(n, nnz)
    print(ia[0:10])
    print(ja[0:10])
    print(a[0:10])
    # plot_spy(n, nnz, ia, ja, a, "spy_plot.png")
