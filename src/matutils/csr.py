import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt


def generate_bta_matrix(
    n: int, b: int, a: int, sparsity: float = 1.0, ensure_diag: bool = True
) -> sp.csr_matrix:
    """Generate a block-tridiagonal with arrowhead matrix.

    Parameters
    ----------
    n : int
        Number of diagonal blocks.
    b : int
        Size of the diagonal blocks.
    a : int
        Size of the arrowhead blocks.
    sparsity : float (default: 1.0)
        Density of the blocks.
    ensure_diag : bool (default: True)
        Ensure that the diagonal elements of the matrix are non-null.
    """
    m = n * b + a

    A = sp.lil_matrix((m, m))    
    for i in range(n):
        # Diagonal block
        A[i * b : (i + 1) * b, i * b : (i + 1) * b] = sp.random(
            b, b, density=sparsity, format="lil"
        )
        if ensure_diag:
            A[i * b : (i + 1) * b, i * b : (i + 1) * b] += np.diag(np.random.rand(b))
        if i < n-1:
            # Off-diagonal blocks
            A[(i + 1) * b : (i + 2) * b, i * b : (i + 1) * b] = sp.random(
                b, b, density=sparsity, format="lil"
            )
            A[i * b : (i + 1) * b, (i + 1) * b : (i + 2) * b] = sp.random(
                b, b, density=sparsity, format="lil"
            )
        # Arrow blocks
        A[-a:, i * b : (i + 1) * b] = sp.random(a, b, density=sparsity, format="lil")
        A[i * b : (i + 1) * b, -a:] = sp.random(b, a, density=sparsity, format="lil")
    # Arrow tip block
    A[-a:, -a:] = np.random.rand(a, a)

    return A.tocsr()


# def generate_ba_matrix(
#     m: int, bwd: int, a: int, sparsity: float = 1.0, ensure_diag: bool = True
# ):
#     """Generate a banded with arrowhead matrix.

#     Parameters
#     ----------
#     m : int
#         Size of the matrix.
#     bwd : int
#         Bandwidth of the matrix.
#     a : int
#         Size of the arrowhead blocks.
#     sparsity : float (default: 1.0)
#         Density of matrix.
#     ensure_diag : bool (default: True)
#         Ensure that the diagonal elements of the matrix are non-null.
#     """

#     A = sp.lil_matrix((m, m))
#     for i in range(m):
#         # Band at row i
#         A[i, max(0, i - bwd) : min(m, i + bwd + 1)] = sp.random(
#             1, 2 * bwd + 1, density=sparsity, format="lil"
#         )
#         if ensure_diag:
#             A[i, i] += np.random.rand()
#         # Arrow block
#         A[-a:, i] = sp.random(a, 1, density=sparsity, format="lil")
#         A[i, -a:] = sp.random(1, a, density=sparsity, format="lil")


def make_diagonaly_dominant(A: sp.csr_matrix):
    m = A.shape[0]
    for i in range(m):
        A[i, i] += np.sum(A[i, :])

def symmetrize(A: sp.csr_matrix):
    A[:] = A + A.T

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
    b = 10
    a = 5

    A = generate_bta_matrix(n, b, a, sparsity=0.1, ensure_diag=True)
    make_diagonaly_dominant(A)
    symmetrize(A)

    plt.spy(A, markersize=1)
    plt.matshow(A.todense())
    plt.show()
