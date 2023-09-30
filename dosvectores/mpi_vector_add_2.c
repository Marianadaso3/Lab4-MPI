#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm);
void Read_n(int* n_p, int* local_n_p, int my_rank, int comm_sz, MPI_Comm comm);
void Allocate_vectors(double** local_x_pp, double** local_y_pp, double** local_z_pp, int local_n, MPI_Comm comm);
void Read_vector(double local_a[], int local_n, int n, char vec_name[], int my_rank, MPI_Comm comm);
void Print_vector(double local_b[], int local_n, int n, char title[], int my_rank, MPI_Comm comm);
void Parallel_vector_sum(double local_x[], double local_y[], double local_z[], int local_n);
double Parallel_dot_product(double local_x[], double local_y[], int local_n);
void Parallel_scalar_product(double local_x[], double local_y[], double scalar, int local_n);

int main(void) {
   int n, local_n;
   int comm_sz, my_rank;
   double *local_x, *local_y, *local_z;
   double dot_product;
   double scalar = 2.0; // Escalar para el producto por escalar

   MPI_Comm comm;
   MPI_Init(NULL, NULL);
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &comm_sz);
   MPI_Comm_rank(comm, &my_rank);

   Read_n(&n, &local_n, my_rank, comm_sz, comm);
   Allocate_vectors(&local_x, &local_y, &local_z, local_n, comm);
   
   Read_vector(local_x, local_n, n, "x", my_rank, comm);
   Read_vector(local_y, local_n, n, "y", my_rank, comm);
   
   // Calcula el producto punto de los vectores locales
   dot_product = Parallel_dot_product(local_x, local_y, local_n);

   // Calcula el producto de un escalar por cada vector
   Parallel_scalar_product(local_x, local_x, scalar, local_n);
   Parallel_scalar_product(local_y, local_y, scalar, local_n);

   // Imprime el resultado del producto punto y los vectores multiplicados por el escalar
   if (my_rank == 0) {
      printf("Producto Punto: %lf\n", dot_product);
      printf("Vector x multiplicado por el escalar:\n");
      Print_vector(local_x, local_n, n, "x is", my_rank, comm);
      printf("Vector y multiplicado por el escalar:\n");
      Print_vector(local_y, local_n, n, "y is", my_rank, comm);
   }

   free(local_x);
   free(local_y);
   free(local_z);

   MPI_Finalize();

   return 0;
}

// ... (las demás funciones se mantienen igual)

// Función para calcular el producto punto de dos vectores locales
double Parallel_dot_product(double local_x[], double local_y[], int local_n) {
   int local_i;
   double local_dot_product = 0.0;

   for (local_i = 0; local_i < local_n; local_i++) {
      local_dot_product += local_x[local_i] * local_y[local_i];
   }

   double global_dot_product;
   MPI_Allreduce(&local_dot_product, &global_dot_product, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   return global_dot_product;
}

// Función para calcular el producto de un escalar por un vector local
void Parallel_scalar_product(double local_x[], double local_y[], double scalar, int local_n) {
   int local_i;

   for (local_i = 0; local_i < local_n; local_i++) {
      local_x[local_i] *= scalar;
      local_y[local_i] *= scalar;
   }
}
