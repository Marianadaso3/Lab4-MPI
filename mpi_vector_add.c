/* File:     mpi_vector_operations.c
 *
 * Purpose:  Implement parallel vector operations using a block
 *           distribution of the vectors. This version also
 *           illustrates the use of MPI_Scatter and MPI_Gather.
 *
 * Compile:  mpicc -g -Wall -o mpi_vector_operations mpi_vector_operations.c
 * Run:      mpiexec -n <comm_sz> ./mpi_vector_operations
 *
 * Input:    The order of the vectors, n, and the vectors x and y
 * Output:   The dot product of vectors x and y, and the scalar product
 *           of a vector with a scalar value.
 *
 * Notes:     
 * 1.  The order of the vectors, n, should be evenly divisible
 *     by comm_sz
 * 2.  DEBUG compile flag.    
 * 3.  This program does fairly extensive error checking. When
 *     an error is detected, a message is printed, and the processes
 *     quit. Errors detected are incorrect values of the vector
 *     order (negative or not evenly divisible by comm_sz), and
 *     malloc failures.
 *
 * IPP:  Section 3.4.6 (pp. 109 and ff.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm);
void Read_n(int* n_p, int* local_n_p, int my_rank, int comm_sz, MPI_Comm comm);
void Allocate_vectors(double** local_x_pp, double** local_y_pp, int local_n, MPI_Comm comm);
void Read_vector(double local_a[], int local_n, int n, char vec_name[], int my_rank, MPI_Comm comm);
double Compute_dot_product(double local_x[], double local_y[], int local_n);
void Scalar_product(double local_x[], double local_result[], int local_n, double scalar);

/*-------------------------------------------------------------------*/
int main(void) {
    int n, local_n;
    int comm_sz, my_rank;
    double *local_x, *local_y;
    MPI_Comm comm;

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    Read_n(&n, &local_n, my_rank, comm_sz, comm);
#ifdef DEBUG
    printf("Proc %d > n = %d, local_n = %d\n", my_rank, n, local_n);
#endif
    Allocate_vectors(&local_x, &local_y, local_n, comm);

    Read_vector(local_x, local_n, n, "x", my_rank, comm);
    Read_vector(local_y, local_n, n, "y", my_rank, comm);

    double dot_product = Compute_dot_product(local_x, local_y, local_n);
    printf("Proc %d > Dot product: %lf\n", my_rank, dot_product);

    double scalar = 2.0;  // Modify this to change the scalar value
    Scalar_product(local_x, local_x, local_n, scalar);

    // Print the result of the scalar product
    printf("Proc %d > Scalar product (x with %lf): ", my_rank, scalar);
    for (int i = 0; i < local_n; i++) {
        printf("%lf ", local_x[i]);
    }
    printf("\n");

    free(local_x);
    free(local_y);

    MPI_Finalize();

    return 0;
}

/*-------------------------------------------------------------------
 * Function:  Check_for_error
 * Purpose:   Check whether any process has found an error. If so,
 *            print a message and terminate all processes. Otherwise,
 *            continue execution.
 * In args:   local_ok:  1 if the calling process has found an error, 0
 *               otherwise
 *            fname:     name of the function calling Check_for_error
 *            message:   message to print if there's an error
 *            comm:      communicator containing processes calling
 *                       Check_for_error: should be MPI_COMM_WORLD.
 *
 * Note:
 *    The communicator containing the processes calling Check_for_error
 *    should be MPI_COMM_WORLD.
 */
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

/*-------------------------------------------------------------------
 * Function:  Read_n
 * Purpose:   Get the order of the vectors from stdin on proc 0 and
 *            broadcast it to other processes.
 * In args:   my_rank:    process rank in communicator
 *            comm_sz:    number of processes in communicator
 *            comm:       communicator containing all the processes
 *                        calling Read_n
 * Out args:  n_p:        global value of n
 *            local_n_p:  local value of n = n/comm_sz
 *
 * Errors:    n should be positive and evenly divisible by comm_sz
 */
void Read_n(int* n_p, int* local_n_p, int my_rank, int comm_sz, MPI_Comm comm) {
    int local_ok = 1;
    char* fname = "Read_n";

    if (my_rank == 0) {
        printf("What's the order of the vectors?\n");
        scanf("%d", n_p);
    }
    MPI_Bcast(n_p, 1, MPI_INT, 0, comm);
    if (*n_p <= 0 || *n_p % comm_sz != 0) local_ok = 0;
    Check_for_error(local_ok, fname,
        "n should be > 0 and evenly divisible by comm_sz", comm);
    *local_n_p = *n_p / comm_sz;
}

/*-------------------------------------------------------------------
 * Function:  Allocate_vectors
 * Purpose:   Allocate storage for x and y
 * In args:   local_n:  the size of the local vectors
 *            comm:     the communicator containing the calling processes
 * Out args:  local_x_pp, local_y_pp:  pointers to memory
 *               blocks to be allocated for local vectors
 *
 * Errors:    One or more of the calls to malloc fails
 */
void Allocate_vectors(double** local_x_pp, double** local_y_pp, int local_n, MPI_Comm comm) {
    int local_ok = 1;
    char* fname = "Allocate_vectors";

    *local_x_pp = malloc(local_n * sizeof(double));
    *local_y_pp = malloc(local_n * sizeof(double));

    if (*local_x_pp == NULL || *local_y_pp == NULL) local_ok = 0;
    Check_for_error(local_ok, fname, "Can't allocate local vector(s)", comm);
}

/*-------------------------------------------------------------------
 * Function:  Read_vector
 * Purpose:   Read a vector from stdin on process 0 and distribute
 *            it among the processes using a block distribution.
 * In args:   local_n:  size of local vectors
 *            n:        size of global vector
 *            vec_name: name of vector being read (e.g., "x")
 *            my_rank:  calling process' rank in comm
 *            comm:     communicator containing calling processes
 * Out arg:   local_a:  local vector read
 *
 * Errors:    If the malloc on process 0 for temporary storage
 *            fails, the program terminates
 *
 * Note: 
 *    This function assumes a block distribution and the order
 *    of the vector is evenly divisible by comm_sz.
 */
void Read_vector(double local_a[], int local_n, int n, char vec_name[], int my_rank, MPI_Comm comm) {
    double* a = NULL;
    int i;
    int local_ok = 1;
    char* fname = "Read_vector";

    if (my_rank == 0) {
        a = malloc(n * sizeof(double));
        if (a == NULL) local_ok = 0;
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", comm);
        printf("Enter the vector %s\n", vec_name);
        for (i = 0; i < n; i++)
            scanf("%lf", &a[i]);
        MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, comm);
        free(a);
    } else {
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", comm);
        MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, comm);
    }
}

/*-------------------------------------------------------------------
 * Function:  Compute_dot_product
 * Purpose:   Compute the dot product of two vectors
 * In args:   local_x:  local storage of the first vector
 *            local_y:  local storage of the second vector
 *            local_n:  the number of components in local_x and local_y
 * Returns:   the dot product of local_x and local_y
 */
double Compute_dot_product(double local_x[], double local_y[], int local_n) {
    double dot_product = 0.0;

    for (int i = 0; i < local_n; i++) {
        dot_product += local_x[i] * local_y[i];
    }

    return dot_product;
}

/*-------------------------------------------------------------------
 * Function:  Scalar_product
 * Purpose:   Compute the scalar product of a vector with a scalar value
 * In/out args:
 *            local_x:      local storage of the vector to be multiplied
 *            local_result: local storage for the result
 *            local_n:      the number of components in local_x and local_result
 *            scalar:       the scalar value
 */
void Scalar_product(double local_x[], double local_result[], int local_n, double scalar) {
    for (int i = 0; i < local_n; i++) {
        local_result[i] = local_x[i] * scalar;
    }
}
