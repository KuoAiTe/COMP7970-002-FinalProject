#include<cstdio>
#include<cstdlib>
#include<ctime>
#include"kmeans.h"

using namespace std;
int main(){
  srand(time(NULL));
  clock_t t = clock();
  SparseMatrix a;
  kmeans b(100,100);
  a.loadFile("./data/edges.csv");
  b.fit(&a);
  b.solve();
  t = clock() - t;
  printf ("It took me %ld clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
}
