#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<vector>
#include<stack>
#include<set>
#include<map>
#include <mpi.h>

#include "par_sparsematrix.h"

using namespace std;

class kmeans{
    
    int k_cluster;
    int max_iter;
    
    SparseMatrix *sparseMatrix;
    
public:
    
    vector< map<int, double> > initializeCentroids();
    kmeans(int _k,int _max);
    
    void fit(SparseMatrix* data);
    void solve(int nprocs, int rank);
    void normalizeCentroids(vector< map<int,double> > &centroids);
    void updateCentroids( vector< map<int,double> > &centroids, int *idx, stack<int> &isolated);
    
    double assignCluster( vector< map<int,double> > &centroids, int *idx, stack<int> &isolated);
    double assignClusterByRank(vector< map<int,double> > &centroids, int * idx, stack<int> &isolated, int rank, int nprocs);
    
};

