#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

void Generate_random_vector(double local_a[], int local_n);
void Print_first_last_elements(double local_b[], int local_n, char title[], int my_rank, MPI_Comm comm);
void Parallel_vector_sum(double local_x[], double local_y[], double local_z[], int local_n);

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

int main(void) {
   int n, local_n;
   int comm_sz, my_rank;
   double *local_x, *local_y, *local_z;
   MPI_Comm comm;

   MPI_Init(NULL, NULL);
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &comm_sz);
   MPI_Comm_rank(comm, &my_rank);

   n = 100000; // valor solicitado por el lab
   local_n = n / comm_sz;

   local_x = (double *)malloc(local_n * sizeof(double));
   local_y = (double *)malloc(local_n * sizeof(double));
   local_z = (double *)malloc(local_n * sizeof(double));

   srand(time(NULL) + my_rank);

   Generate_random_vector(local_x, local_n);
   Generate_random_vector(local_y, local_n);

   Parallel_vector_sum(local_x, local_y, local_z, local_n);

   Print_first_last_elements(local_x, local_n, "Vector local x", my_rank, comm);
   Print_first_last_elements(local_y, local_n, "Vector local y", my_rank, comm);
   Print_first_last_elements(local_z, local_n, "Vector local suma", my_rank, comm);

   free(local_x);
   free(local_y);
   free(local_z);

   MPI_Finalize();

   return 0;
}

void Generate_random_vector(double local_a[], int local_n) {
   for (int i = 0; i < local_n; i++) {
      local_a[i] = (double)rand() / RAND_MAX; // Genera números aleatorios entre 0 y 1
   }
}

void Print_first_last_elements(double local_b[], int local_n, char title[], int my_rank, MPI_Comm comm) {
   double *b = NULL;
   int n;
   int local_ok = 1;
   char *fname = "Print_vector";

   MPI_Allreduce(&local_n, &n, 1, MPI_INT, MPI_SUM, comm);

   if (my_rank == 0) {
      b = (double *)malloc(n * sizeof(double));
      if (b == NULL) local_ok = 0;
      MPI_Gather(local_b, local_n, MPI_DOUBLE, b, local_n, MPI_DOUBLE, 0, comm);
      printf("%s:\n", title);
      for (int i = 0; i < 10; i++) {
         printf("%.6f ", b[i]);
      }
      printf("... ");
      for (int i = n - 10; i < n; i++) {
         printf("%.6f ", b[i]);
      }
      printf("\n");
      free(b);
   } else {
      MPI_Gather(local_b, local_n, MPI_DOUBLE, b, local_n, MPI_DOUBLE, 0, comm);
   }

   Check_for_error(local_ok, fname, "Can't allocate temporary vector", comm);
}

void Parallel_vector_sum(double local_x[], double local_y[], double local_z[], int local_n) {
   for (int local_i = 0; local_i < local_n; local_i++) {
      local_z[local_i] = local_x[local_i] + local_y[local_i];
   }
}
