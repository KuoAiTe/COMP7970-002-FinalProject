#include<cstdio>
#include<cstdlib>
#include<ctime>
#include <mpi.h>
#include"par_kmeans.h"

using namespace std;

int main(int argc, const char *argv[]) {
    
    int rank,nprocs;
    char **pargv;
    
    MPI_Init(&argc, &pargv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    srand(time(NULL));
    clock_t t = clock();
    SparseMatrix a;
    kmeans b(100,100);
    a.loadFile("./data/edges.csv", rank);
    b.fit(&a);
    b.solve(nprocs, rank);
    t = clock() - t;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if(rank == 0) {
        dbprintf ("\n\nIt took me %ld clicks (%f seconds).\n\n",t,((float)t)/CLOCKS_PER_SEC);
    }
    
    MPI_Finalize();
    
}
