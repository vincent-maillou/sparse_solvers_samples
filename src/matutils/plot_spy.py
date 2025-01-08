import numpy as np
import matplotlib.pyplot as plt

def read_matrix_from_file(filename):
    with open(filename, "r") as file:
        n, nnz = map(int, file.readline().split())
        ia = list(map(int, file.readline().split()))
        ja = list(map(int, file.readline().split()))
        a = list(map(float, file.readline().split()))
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
    n, nnz, ia, ja, a = read_matrix_from_file("matrix_data.txt")
    plot_spy(n, nnz, ia, ja, a, "spy_plot.png")
