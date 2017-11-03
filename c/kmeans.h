#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<vector>
#include<stack>
#include<set>
#include<map>
using namespace std;
#include "SparseMatrix.h"
class kmeans{
  int k_cluster;
  int max_iter;
  SparseMatrix *sparseMatrix;
  public:
  kmeans(int _k,int _max);
  void fit(SparseMatrix* data);
  void solve();
  vector< map<int, double> > initializeCentroids();
  void normalizeCentroids(vector< map<int,double> > &centroids);
  double assignCluster( vector< map<int,double> > &centroids, int *idx, stack<int> &isolated);
  void updateCentroids( vector< map<int,double> > &centroids, int *idx, stack<int> &isolated);
};
