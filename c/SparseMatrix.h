#include<cstdio>
#include<cstdlib>
#include<vector>
#include<set>
#include<utility>
#include<stack>
#include<map>
using namespace std;
class SparseMatrix{
private:
  vector< pair<int,int> > getFeaturesByInstance; // size of instance
  vector< vector<int> > relevantInstance;
  int m_size;
  int n_size;
public:
  SparseMatrix();
  int getInstanceSize();
  int getNodeSize();
  map<int,double> randomlyPickAnInstance();
  map<int,double> pickInstanceFromIsolated(stack <int> &isolated);
  map<int,double> getInstanceFromMatrix(int instanceIndex);
  set<int> getRelevantInstanceSetByFeatureIndexAndCentroid(map<int,double> &centroid);
  double calculateSimilarity(int instanceIndex, map<int,double> &centroid);
  void updateCentroid(int instanceIndex, map<int,double> &centroid);
  bool loadFile(const char* filename);
};
