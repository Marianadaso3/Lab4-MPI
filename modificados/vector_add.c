#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm);
void Read_n(int* n_p, int* local_n_p, int my_rank, int comm_sz, MPI_Comm comm);
void Allocate_vectors(double** local_x_pp, double** local_y_pp, double** local_z_pp, int local_n, MPI_Comm comm);
void Generate_random_vector(double local_a[], int local_n);
void Print_first_last_elements(double local_b[], int local_n, char vec_name[], int my_rank, MPI_Comm comm);
double Parallel_vector_dot_product(double local_x[], double local_y[], int local_n);
double dot_product_result = Parallel_vector_dot_product(local_x, local_y, local_n);
void Parallel_scalar_vector_product(double local_x[], double local_y[], double scalar, int local_n);

int main(void) {
    int n, local_n;
    int comm_sz, my_rank;
    double *local_x, *local_y, *local_z;
    double scalar = 250.0; // Escalar para el producto escalar
    MPI_Comm comm;
    clock_t start_time, end_time; 

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    Read_n(&n, &local_n, my_rank, comm_sz, comm);

    Allocate_vectors(&local_x, &local_y, &local_z, local_n, comm);

    Generate_random_vector(local_x, local_n);
    Print_first_last_elements(local_x, local_n, "x", my_rank, comm);

    Generate_random_vector(local_y, local_n);
    Print_first_last_elements(local_y, local_n, "y", my_rank, comm);

    start_time = clock(); 
    Parallel_vector_dot_product(local_x, local_y, local_n);
    end_time = clock(); 

    double total_time_dot_product = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    start_time = clock(); 
    Parallel_scalar_vector_product(local_x, local_y, scalar, local_n);
    end_time = clock(); 

    double total_time_scalar_product = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    //Print_first_last_elements(local_z, local_n, "Dot Product Result (local)", my_rank, comm);

    Parallel_scalar_vector_product(local_x, local_y, scalar, local_n);
    Print_first_last_elements(local_y, local_n, "Scalar Vector Product Result (local)", my_rank, comm);

   if (my_rank == 0) {
      printf("Scalar: %.2f\n", scalar);
      printf("Total Time for Dot Product: %.6f seconds\n", total_time_dot_product); 
      printf("Total Time for Scalar Vector Product: %.6f seconds\n", total_time_scalar_product); 
   }

    free(local_x);
    free(local_y);
    free(local_z);

    MPI_Finalize();

    return 0;
}

void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm) {
    int ok;

    MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, comm);
    if (ok == 0) {
        int my_rank;
        MPI_Comm_rank(comm, &my_rank);
        if (my_rank == 0) {
            fprintf(stderr, "Proc %d > In %s, %s\n", my_rank, fname, message);
            fflush(stderr);
        }
        MPI_Finalize();
        exit(-1);
    }
}

void Read_n(int* n_p, int* local_n_p, int my_rank, int comm_sz, MPI_Comm comm) {
    int local_ok = 1;
    char *fname = "Read_n";

    if (my_rank == 0) {
        *n_p = 100000; // Set the order of the vectors to at least 100,000
    }
    MPI_Bcast(n_p, 1, MPI_INT, 0, comm);
    if (*n_p <= 0 || *n_p % comm_sz != 0) local_ok = 0;
    Check_for_error(local_ok, fname, "n should be > 0 and evenly divisible by comm_sz", comm);
    *local_n_p = *n_p / comm_sz;
}

void Allocate_vectors(double** local_x_pp, double** local_y_pp, double** local_z_pp, int local_n, MPI_Comm comm) {
    int local_ok = 1;
    char* fname = "Allocate_vectors";

    *local_x_pp = malloc(local_n * sizeof(double));
    *local_y_pp = malloc(local_n * sizeof(double));
    *local_z_pp = malloc(local_n * sizeof(double));

    if (*local_x_pp == NULL || *local_y_pp == NULL || *local_z_pp == NULL) local_ok = 0;
    Check_for_error(local_ok, fname, "Can't allocate local vector(s)", comm);
}

void Generate_random_vector(double local_a[], int local_n) {
    int i;

    srand(time(NULL)); // Seed the random number generator
    for (i = 0; i < local_n; i++) {
        local_a[i] = ((double)rand() / RAND_MAX) * 100.0; // Generate random values between 0 and 100
    }
}

void Print_first_last_elements(double local_b[], int local_n, char vec_name[], int my_rank, MPI_Comm comm) {
    double* b = NULL;
    int i;
    int local_ok = 1;
    char* fname = "Print_vector";

    if (my_rank == 0) {
        b = malloc(local_n * sizeof(double));
        if (b == NULL) local_ok = 0;
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", comm);
        MPI_Gather(local_b, local_n, MPI_DOUBLE, b, local_n, MPI_DOUBLE, 0, comm);
        printf("%s (First and Last 10 elements):\n", vec_name);
        for (i = 0; i < 10; i++) {
            printf("%f ", b[i]);
        }
        printf("... ");
        for (i = local_n - 10; i < local_n; i++) {
            printf("%f ", b[i]);
        }
        printf("\n");
        free(b);
    } else {
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", comm);
        MPI_Gather(local_b, local_n, MPI_DOUBLE, b, local_n, MPI_DOUBLE, 0, comm);
    }
}

double Parallel_vector_dot_product(double local_x[], double local_y[], int local_n) {
    double local_dot_product = 0.0;
    for (int local_i = 0; local_i < local_n; local_i++) {
        local_dot_product += local_x[local_i] * local_y[local_i];
    }
    double global_dot_product;
    MPI_Allreduce(&local_dot_product, &global_dot_product, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_dot_product;
}

void Parallel_scalar_vector_product(double local_x[], double local_y[], double scalar, int local_n) {
    for (int local_i = 0; local_i < local_n; local_i++) {
        local_y[local_i] = scalar * local_x[local_i];
    }
}