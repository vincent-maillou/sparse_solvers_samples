#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"
#include "dmumps_c.h"

#define JOB_INIT -1
#define JOB_END -2
#define USE_COMM_WORLD -987654

// Define the ICNTL macro to access the icntl array
#define ICNTL(I) icntl[(I)-1]

void load_sym_coo(const char *filename, MUMPS_INT *n, MUMPS_INT8 *nnz, MUMPS_INT **row, MUMPS_INT **col, double **data);

int main(int argc, char **argv) {
    DMUMPS_STRUC_C id;
    MUMPS_INT n;
    MUMPS_INT8 nnz;
    MUMPS_INT *row, *col;
    double *data;
    double *rhs;
    int myid, ierr;

    double start_time = 0.0;
    double end_time = 0.0;
    double elapsed_time = 0.0;

    // Initialize MPI
    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // Load the CSC matrix
    // load_sym_coo("/capstor/scratch/cscs/vmaillou/data/bta_dataset/Qxy_ns2865_nt365_nss0_nb4_n1045729_mumps.dat", &n, &nnz, &row, &col, &data);
    load_sym_coo("/capstor/scratch/cscs/vmaillou/data/bta_dataset/Qxy_ns42_nt3_nss0_nb2_n128_mumps.dat", &n, &nnz, &row, &col, &data);

    // Print the first n_elem elements of the CSC arrays
    int n_elem = 10;
    if (myid == 0) {
        printf("row: ");
        for (size_t i = 0; i < n_elem; i++) {
            printf("%d ", row[i]);
        }
        printf("\n");

        printf("col: ");
        for (size_t i = 0; i < n_elem; i++) {
            printf("%d ", col[i]);
        }
        printf("\n");

        printf("data: ");
        for (size_t i = 0; i < n_elem; i++) {
            printf("%f ", data[i]);
        }
        printf("\n");
    }

    // Print all the diagonal elements
    /* if (myid == 0) {
        printf("Diagonal elements: ");
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < nnz; j++) {
                if (row[j] == i && col[j] == i) {
                    printf("%f ", data[j]);
                    break;
                }
            }
        }
        printf("\n");
    } */

    
    // Allocate memory for the right-hand-side vector (identity matrix)
    rhs = (double *)malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++) {
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
        id.irn = row;
        id.jcn = col;
        id.a = data;
        id.rhs = rhs;
    }
    
    // Set MUMPS parameters
    id.ICNTL(1) = 6; // Error messages (6 default std::out)
    id.ICNTL(2) = 6; // Warning messages (6 default std::out)
    id.ICNTL(3) = 6; // Output stream for global information
    // id.ICNTL(4) = 4; // Output stream for statistics
    id.ICNTL(13) = 0; // Cholesky instead of LDL^T
    // id.ICNTL(14) = 100; // Increase workspace by 20%
    // id.ICNTL(22) = 1; // Enable out-of-core option
    // id.ICNTL(30) = 1; // Enable selected inversion
    id.ICNTL(5) = 0;
    id.ICNTL(18) = 0;

    // id.ICNTL(7) = 5; // Ordering method
    
    
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
    
    
    /*
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
    free(row);
    free(col);
    free(data);
    free(rhs);

    // Finalize MPI
    ierr = MPI_Finalize();

    return 0;
}


void load_sym_coo(const char *filename, MUMPS_INT *n, MUMPS_INT8 *nnz, MUMPS_INT **row, MUMPS_INT **col, double **data) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    int32_t temp_n, temp_nnz;

    // Read the number of rows/columns and number of non-zero elements
    fread(&temp_n, sizeof(int32_t), 1, file);
    fread(&temp_n, sizeof(int32_t), 1, file); // Assuming the file repeats the dimension
    fread(&temp_nnz, sizeof(int32_t), 1, file);

    // Cast to MUMPS datatype
    *n = (MUMPS_INT)temp_n;
    *nnz = (MUMPS_INT8)temp_nnz;

    // Allocate memory for temporary int32_t arrays
    int32_t *temp_row = (int32_t *)malloc(*nnz * sizeof(int32_t));
    int32_t *temp_col = (int32_t *)malloc(*nnz * sizeof(int32_t));

    // Allocate memory for MUMPS_INT arrays
    *row = (MUMPS_INT *)malloc(*nnz * sizeof(MUMPS_INT));
    *col = (MUMPS_INT *)malloc(*nnz * sizeof(MUMPS_INT));
    *data = (double *)malloc(*nnz * sizeof(double));

    // Read the COO arrays from the file into temporary arrays
    fread(temp_row, sizeof(int32_t), *nnz, file);
    fread(temp_col, sizeof(int32_t), *nnz, file);
    fread(*data, sizeof(double), *nnz, file);

    // Convert temporary arrays to MUMPS_INT arrays
    for (MUMPS_INT8 i = 0; i < *nnz; i++) {
        // Also convert to 0-based to 1-based indexing
        (*row)[i] = (MUMPS_INT)(temp_row[i]+1);
        (*col)[i] = (MUMPS_INT)(temp_col[i]+1);
    }

    // Free temporary arrays
    free(temp_row);
    free(temp_col);

    // Calculate and print the allocated memory for the matrix
    size_t row_size = *nnz * sizeof(MUMPS_INT);
    size_t col_size = *nnz * sizeof(MUMPS_INT);
    size_t data_size = *nnz * sizeof(double);
    size_t total_size = row_size + col_size + data_size;

    printf("Allocated memory for matrix dimensions: n = %d, nnz = %lld\n", *n, *nnz);
    printf("  row: %zu bytes\n", row_size);
    printf("  col: %zu bytes\n", col_size);
    printf("  data: %zu bytes\n", data_size);
    printf("  Total: %zu bytes (%.2f MB)\n", total_size, total_size / (1024.0 * 1024.0));


    fclose(file);
}
