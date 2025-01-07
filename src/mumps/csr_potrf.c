#include <stdio.h>
#include <stdlib.h>
#include "dmumps_c.h"

#define JOB_INIT -1
#define JOB_END -2
#define USE_COMM_WORLD -987654

void load_csr_matrix(const char *filename, int *n, int *nnz, int **ia, int **ja, double **a) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Read the number of rows/columns and number of non-zero elements
    fread(n, sizeof(int), 1, file);
    fread(nnz, sizeof(int), 1, file);

    // Allocate memory for CSR arrays
    *ia = (int *)malloc((*n + 1) * sizeof(int));
    *ja = (int *)malloc(*nnz * sizeof(int));
    *a = (double *)malloc(*nnz * sizeof(double));

    // Read the CSR arrays from the file
    fread(*ia, sizeof(int), *n + 1, file);
    fread(*ja, sizeof(int), *nnz, file);
    fread(*a, sizeof(double), *nnz, file);

    fclose(file);
}

int main() {
    DMUMPS_STRUC_C id;
    int n, nnz;
    int *ia, *ja;
    double *a;

    // Load the CSR matrix
    load_csr_matrix("matrix.csr", &n, &nnz, &ia, &ja, &a);

    // Initialize MUMPS
    id.job = JOB_INIT;
    id.par = 1;
    id.sym = 1; // Symmetric positive definite matrix
    id.comm_fortran = USE_COMM_WORLD;
    dmumps_c(&id);

    // Define the problem on the host
    id.n = n;
    id.nz = nnz;
    id.irn = ia;
    id.jcn = ja;
    id.a = a;

    // Perform Cholesky decomposition
    id.job = 4; // Analysis + Factorization
    dmumps_c(&id);

    // Check for errors
    if (id.infog[0] != 0) {
        printf("MUMPS error: %d\n", id.infog[0]);
    } else {
        printf("Cholesky decomposition successful.\n");
    }

    // Terminate MUMPS
    id.job = JOB_END;
    dmumps_c(&id);

    // Free allocated memory
    free(ia);
    free(ja);
    free(a);

    return 0;
}