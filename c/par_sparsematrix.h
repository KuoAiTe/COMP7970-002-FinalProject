
#include<cstdio>
#include<cstdlib>
#include<vector>
#include<set>
#include<utility>
#include<stack>
#include<map>

// print macro for parallel debugging...

#ifdef dbprintf
#undef dbprintf(msg, args...)
#endif
#define dbprintf(msg, args...) \
do { \
int rank; \
MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
printf("%i:%d:%s(): ", rank, __LINE__, __func__); \
printf(msg, ##args); \
} while(0)

#define DEBUG 1  // remove to disable verbose output

using namespace std;

class SparseMatrix {
    
    // edge: connection between a pair of features, represented by two points (x, y)
    // instance: refers to the record \ line number of an edge (x,y) in the data file
    // feature: the value of either x or y in the edge pair, aka node
    
    //    8---7  4
    //   / \ /\ /|\ <---- edge (x=4, y=3)
    //   9--5--1---3
    //    \ | /  |/
    //     \|/   2  <---- node (2)
    //      6
    
    // note that a node can be associated with one or more edges
    
private:
    
    // the following variable looks like a method \ function name, but is vector of x,y points
    // that allows retrieval of an edge (x,y) by its instance, or record number...
    
    vector< pair<int,int> > getFeaturesByInstance;
    
    // relevant instance returns the no1de instances (edges) associated with a
    // given feature (node). i.e...
    //
    // relevantInstance[6] = { 3,4,5 }
    //
    // the instance number can then be provided to getFeaturesByInstance
    // to retrieve the connection between the pair of nodes (edge) if needed
    
    vector< vector<int> > relevantInstance;
    
    int m_size;  // records the highest value of feature in input
    int n_size;  // full size of input, count of edges
    int n_procs;
    
public:
    
    SparseMatrix();
    
    int getRankInstanceSize();
    int getInstanceSize();
    int getNodeSize();
    
    map<int,double> randomlyPickAnInstance();
    map<int,double> pickInstanceFromIsolated(stack <int> &isolated);
    map<int,double> getInstanceFromMatrix(int instanceIndex);
    
    int getRelevantInstanceSetByFeatureIndexAndCentroid(map<int,double> &centroid, int *relevantInstanceSet);
    
    double calculateSimilarity(int instanceIndex, map<int,double> &centroid);
    
    void updateCentroid(int instanceIndex, map<int,double> &centroid);
    
    bool loadFile(const char* filename, int rank);
    
};

