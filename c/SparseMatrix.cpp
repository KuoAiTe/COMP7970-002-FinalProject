#include<vector>
#include "SparseMatrix.h"
using namespace std;
#ifndef SPARSE_MATRIX
#define SPARSE_MATRIX
SparseMatrix::SparseMatrix(){
}
map<int,double> SparseMatrix::pickInstanceFromIsolated(stack <int> &isolated){
    if (!isolated.empty()){
      int idx = isolated.top();
      isolated.pop();
      return getInstanceFromMatrix(idx);
    }else
      return randomlyPickAnInstance();
}
void SparseMatrix::updateCentroid(int instanceIndex, map<int,double> &centroid){
    int featureIndex_1 = getFeaturesByInstance[instanceIndex].first;
    int featureIndex_2 = getFeaturesByInstance[instanceIndex].second;
    centroid[featureIndex_1] = ( centroid.find(featureIndex_1) != centroid.end() )?centroid[featureIndex_1] + 1:1;
    centroid[featureIndex_2] = ( centroid.find(featureIndex_2) != centroid.end() )?centroid[featureIndex_2] + 1:1;
}
map<int,double> SparseMatrix::randomlyPickAnInstance(){
    int instanceIndex = rand()%n_size;
    map<int,double> instance = getInstanceFromMatrix(instanceIndex);
    return instance;
}
map<int,double> SparseMatrix::getInstanceFromMatrix(int instanceIndex){
    map<int,double> instance;
    int featureIndex_1 = getFeaturesByInstance[instanceIndex].first;
    int featureIndex_2 = getFeaturesByInstance[instanceIndex].second;
    instance[featureIndex_1] = 1;
    instance[featureIndex_2] = 1;
    return instance;
}
set<int> SparseMatrix::getRelevantInstanceSetByFeatureIndexAndCentroid(map<int,double> &centroid){
    set<int> relevantInstanceSet;
    int featureIndex, instanceIndex;
    for(map<int,double>::iterator it = centroid.begin();  it != centroid.end(); it++){
        featureIndex = it->first;
        for(int i=0;i<relevantInstance[featureIndex].size();i++){
            instanceIndex = relevantInstance[featureIndex][i];
            relevantInstanceSet.insert(instanceIndex);
        }
    }
    return relevantInstanceSet;
}
double SparseMatrix::calculateSimilarity(int instanceIndex, map<int,double> &centroid){
    double similarity = 0;
    int featureIndex_1 = getFeaturesByInstance[instanceIndex].first;
    int featureIndex_2 = getFeaturesByInstance[instanceIndex].second;
    similarity += ( centroid.find(featureIndex_1) != centroid.end() )?centroid[featureIndex_1]:0;
    similarity += ( centroid.find(featureIndex_2) != centroid.end() )?centroid[featureIndex_2]:0;
    return similarity;
}
bool SparseMatrix::loadFile(const char* filename){
    FILE *fp = fopen(filename, "r");
    if (fp) {
      int node_1,node_2;
      m_size = n_size = 0;
      getFeaturesByInstance.clear();
      relevantInstance.clear();
      // build m -> nodes n -> edges matrix
      // m = nodes, n = edges_size
      while (fscanf(fp, "%d,%d", &node_1,&node_2) !=EOF ){
        m_size = (node_1>m_size)?node_1:m_size;
        m_size = (node_2>m_size)?node_2:m_size;
        getFeaturesByInstance.push_back(make_pair(node_1,node_2));
        n_size++;
      }

      // build feature to instance
      for(int i=0;i<m_size;i++)
        relevantInstance.push_back(vector<int>());
      for(int instanceIndex=0;instanceIndex<n_size;instanceIndex++){
        int featureIndex_1 = getFeaturesByInstance[instanceIndex].first;
        int featureIndex_2 = getFeaturesByInstance[instanceIndex].second;
        relevantInstance[featureIndex_1].push_back(instanceIndex);
        relevantInstance[featureIndex_2].push_back(instanceIndex);
      }
      fclose(fp);
      return true;
    }
    return false;
}
int SparseMatrix::getNodeSize(){
  return m_size;
}
int SparseMatrix::getInstanceSize(){
  return n_size;
}
#endif
