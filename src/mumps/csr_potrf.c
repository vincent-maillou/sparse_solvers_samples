#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"
#include "dmumps_c.h"

#define JOB_INIT -1
#define JOB_END -2
#define USE_COMM_WORLD -987654

// Define the ICNTL macro to access the icntl array
#define ICNTL(I) icntl[(I)-1]

void load_sym_csc(const char *filename, int32_t *n, int32_t *nnz, int32_t **ia, int32_t **ja, double **a);

int main(int argc, char **argv) {
    DMUMPS_STRUC_C id;
    int32_t n, nnz;
    int32_t *ia, *ja;
    double *a;
    double *rhs;
    int myid, ierr;

    double start_time = 0.0;
    double end_time = 0.0;
    double elapsed_time = 0.0;

    // Initialize MPI
    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // Load the CSC matrix
    load_sym_csc("/capstor/scratch/cscs/vmaillou/data/bta_dataset/Qxy_ns2865_nt365_nss0_nb4_n1045729.dat", &n, &nnz, &ia, &ja, &a);

    // Debug prints
    if (myid == 0) {
        printf("Matrix dimensions: n = %d, nnz = %d\n", n, nnz);
    }

    // Print the first 10 elements of the CSC arrays
    if (myid == 0) {
        printf("ia: ");
        for (int32_t i = 0; i < 10; i++) {
            printf("%d ", ia[i]);
        }
        printf("\n");

        printf("ja: ");
        for (int32_t i = 0; i < 10; i++) {
            printf("%d ", ja[i]);
        }
        printf("\n");

        printf("a: ");
        for (int32_t i = 0; i < 10; i++) {
            printf("%f ", a[i]);
        }
        printf("\n");
    }

    // Allocate memory for the right-hand-side vector (identity matrix)
    rhs = (double *)malloc(n * sizeof(double));
    for (int32_t i = 0; i < n; i++) {
        rhs[i] = 1.0; // Diagonal ones
    }

    // Initialize MUMPS
    id.comm_fortran = USE_COMM_WORLD;
    id.par = 1;
    id.sym = 1; // Symmetric positive definite matrix
    id.job = JOB_INIT;
    dmumps_c(&id);

    // Define the problem on the host
    if (myid == 0) {
        id.n = n;
        id.nnz = nnz;
        id.irn = ia;
        id.jcn = ja;
        id.a = a;
        id.rhs = rhs;
    }
    
    // Set MUMPS parameters
    id.ICNTL(1) = 6; // Output stream for error messages
    id.ICNTL(2) = 6; // Output stream for diagnostic printing
    id.ICNTL(3) = 6; // Output stream for global information
    id.ICNTL(4) = 4; // Output stream for statistics
    // id.ICNTL(14) = 100; // Increase workspace by 20%
    // id.ICNTL(22) = 1; // Enable out-of-core option
    // id.ICNTL(30) = 1; // Enable selected inversion
    
    /*
    // Ordering phase
    printf("Start ordering phase on process %d\n", myid);
    start_time = MPI_Wtime(); // Start timing
    id.job = 1;
    dmumps_c(&id);
    end_time = MPI_Wtime(); // End timing
    elapsed_time = end_time - start_time; // Calculate elapsed time
    if (id.infog[0] < 0) {
        printf(" (PROC %d) ERROR RETURN: \tINFOG(1)= %d\n\t\t\t\tINFOG(2)= %d\n",
               myid, id.infog[0], id.infog[1]);
        MPI_Abort(MPI_COMM_WORLD, id.infog[0]);
    } else {
        // Print estimated workspace requirements
        if (myid == 0) {
            printf("Estimated working space for factorization phase:\n");
            printf("  Double precision words: %d\n", id.infog[15]);
            printf("  Integer words: %d\n", id.infog[16]);
        }
    }
    printf("Ordering phase completed on process %d in %f seconds\n", myid, elapsed_time);

    // Factorization phase
    printf("Start factorization phase on process %d\n", myid);
    start_time = MPI_Wtime(); // Start timing
    id.job = 2;
    dmumps_c(&id);
    end_time = MPI_Wtime(); // End timing
    elapsed_time = end_time - start_time; // Calculate elapsed time
    if (id.infog[0] < 0) {
        printf(" (PROC %d) ERROR RETURN: \tINFOG(1)= %d\n\t\t\t\tINFOG(2)= %d\n",
               myid, id.infog[0], id.infog[1]);
        MPI_Abort(MPI_COMM_WORLD, id.infog[0]);
    }
    printf("Factorization phase completed on process %d in %f seconds\n", myid, elapsed_time);
    
    // Solve phase
    printf("Start solving phase on process %d\n", myid);
    id.job = 3;
    dmumps_c(&id);
    if (id.infog[0] < 0) {
        printf(" (PROC %d) ERROR RETURN: \tINFOG(1)= %d\n\t\t\t\tINFOG(2)= %d\n",
               myid, id.infog[0], id.infog[1]);
        MPI_Abort(MPI_COMM_WORLD, id.infog[0]);
    }

    // Print the selected entries of the inverse
    if (myid == 0) {
        printf("Selected entries of the inverse:\n");
        for (int32_t i = 0; i < n; i++) {
            printf("%f\n", rhs[i]);
        }
    }
    */

    // Terminate MUMPS
    id.job = JOB_END;
    dmumps_c(&id);

    // Free allocated memory
    free(ia);
    free(ja);
    free(a);
    free(rhs);

    // Finalize MPI
    ierr = MPI_Finalize();

    return 0;
}


void load_sym_csc(const char *filename, int32_t *n, int32_t *nnz, int32_t **ia, int32_t **ja, double **a) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Read the number of rows/columns and number of non-zero elements
    fscanf(file, "%d", n);
    fscanf(file, "%d", n); // Assuming the file repeats the dimension
    fscanf(file, "%d", nnz);

    // Allocate memory for CSC arrays
    *ia = (int32_t *)malloc((*n + 1) * sizeof(int32_t));
    *ja = (int32_t *)malloc(*nnz * sizeof(int32_t));
    *a = (double *)malloc(*nnz * sizeof(double));

    // Calculate and print the allocated memory for the matrix
    int32_t ia_size = (*n + 1) * sizeof(int32_t);
    int32_t ja_size = *nnz * sizeof(int32_t);
    int32_t a_size = *nnz * sizeof(double);
    int32_t total_size = ia_size + ja_size + a_size;

    printf("Allocated memory for matrix:\n");
    printf("  ia: %zu bytes\n", ia_size);
    printf("  ja: %zu bytes\n", ja_size);
    printf("  a: %zu bytes\n", a_size);
    printf("  Total: %zu bytes (%.2f MB)\n", total_size, total_size / (1024.0 * 1024.0));

    // Read the CSC arrays from the file
    for (int32_t i = 0; i < *nnz; i++) {
        fscanf(file, "%d", &(*ja)[i]);
    }

    for (int32_t i = 0; i < *n + 1; i++) {
        fscanf(file, "%d", &(*ia)[i]);
    }

    for (int32_t i = 0; i < *nnz; i++) {
        fscanf(file, "%lf", &(*a)[i]);
    }

    fclose(file);
}
