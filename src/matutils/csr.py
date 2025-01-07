import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt

def generate_bta_matrix(n: int, b: int, a: int):
    m = n * b + a

    A = sp.lil_matrix((m, m))    
    for i in range(n):
        # Diagonal block
        A[i*b:(i+1)*b, i*b:(i+1)*b] = np.random.rand(b, b)
        if i < n-1:
            # Off-diagonal blocks
            A[(i+1)*b:(i+2)*b, i*b:(i+1)*b] = np.random.rand(b, b)
            A[i*b:(i+1)*b, (i+1)*b:(i+2)*b] = np.random.rand(b, b)
        # Arrow blocks
        A[-a:, i*b:(i+1)*b] = np.random.rand(a, b)
        A[i*b:(i+1)*b, -a:] = np.random.rand(b, a)
    # Arrow tip block
    A[-a:, -a:] = np.random.rand(a, a)

    return A.to_csr()

def make_diagonaly_dominant(A: sp.csr_matrix):
    m = A.shape[0]
    for i in range(m):
        A[i, i] += np.sum(A[i, :])
    return A

def symmetrize(A: sp.csr_matrix):
    return A + A.T

def save_csr_matrix(filename: str, A: sp.csr_matrix):
    n = A.shape[0]
    nnz = A.nnz
    ia = A.indptr.astype(np.int32)
    ja = A.indices.astype(np.int32)
    a = A.data.astype(np.float64)

    with open(filename, 'wb') as f:
        f.write(n.to_bytes(4, byteorder='little'))
        f.write(nnz.to_bytes(4, byteorder='little'))
        f.write(ia.tobytes())
        f.write(ja.tobytes())
        f.write(a.tobytes())

if __name__ == "__main__":
    n = 10
    b = 3
    a = 2

    A = generate_bta_matrix(n, b, a)
    
    plt.spy(A, markersize=1)
    plt.show()