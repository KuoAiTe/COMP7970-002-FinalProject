#include "kmeans.h"
#include <cmath>
using namespace std;
#ifndef K_MEANS
#define K_MEANS
kmeans::kmeans(int _k,int _max){
    k_cluster = _k;
    max_iter = _max;
}
void kmeans::fit(SparseMatrix* _sparseMatrix){
    sparseMatrix = _sparseMatrix;
}

void kmeans::updateCentroids( vector< map<int,double> > &centroids, int *idx, stack<int> &isolated){
    int *numCluster = new int[k_cluster];
    memset(numCluster, 0, sizeof(int) * k_cluster);
    for (int i=0; i< centroids.size(); i++)
      centroids[i].clear();
    for (int i=0; i< sparseMatrix->getInstanceSize(); i++){
        numCluster[idx[i]]++;
        sparseMatrix->updateCentroid(i, centroids[idx[i]]);
    }
    for (int i=0; i<centroids.size(); i++){
        printf("numCluster[%d] = %d\n", i, numCluster[i]);
        if (numCluster[i] == 0 ){
          centroids[i] = sparseMatrix->pickInstanceFromIsolated(isolated);
        }
    }
    normalizeCentroids(centroids);
}
void kmeans::normalizeCentroids(vector< map<int,double> > &centroids){
    for (int i=0; i<centroids.size(); i++){
        map<int,double> &instance = centroids[i];
        double length = 0;
        for (map<int,double>::iterator it = instance.begin(); it!=instance.end(); it++){
            length += it->second * it->second;
        }
        if (length > 0){
            length = 1/sqrt(length);
            for (map<int,double>::iterator it = instance.begin(); it!=instance.end(); it++){
                instance[it->first] = it->second * length;
            }
        }
    }

}

double kmeans::assignCluster( vector< map<int,double> > &centroids, int * idx, stack<int> &isolated){
    set<int>::iterator it;
    int instanceSize = sparseMatrix->getInstanceSize();
    double * maxSimilarity = new double[instanceSize];
    memset(idx, 0, sizeof(int) * instanceSize);
    memset(maxSimilarity, 0, sizeof(double)*instanceSize);
    double Similarity;
    int instanceIndex;
    set <int> RelevantInstanceSet;
    for (int i=0; i<centroids.size(); i++){
      RelevantInstanceSet = sparseMatrix->getRelevantInstanceSetByFeatureIndexAndCentroid(centroids[i]);
      for(it = RelevantInstanceSet.begin(); it!=RelevantInstanceSet.end(); it++){
        instanceIndex = *it;
        Similarity = sparseMatrix->calculateSimilarity(instanceIndex, centroids[i]);
        if (Similarity > maxSimilarity[instanceIndex]){
          maxSimilarity[instanceIndex] = Similarity;
          idx[instanceIndex] = i;
        }
      }
      RelevantInstanceSet.clear();
    }
    while(!isolated.empty())
      isolated.pop();

    double obj = 0;
    for (int i=0; i<instanceSize; i++){
      obj += maxSimilarity[i];
      if (maxSimilarity[i] < 1e-9)
        isolated.push(i);
    }
    return obj;
}
void kmeans::solve(){
    vector< map<int, double> > centroids = initializeCentroids();
    normalizeCentroids(centroids);
    int instanceSize = sparseMatrix->getInstanceSize();
    int * idx = new int[instanceSize];
    double previous_objectiveValue=0, objectiveValue, objectiveValue_difference,TOL =0.001;
    stack<int> isolated = stack<int>();
    printf("Start Clustering\n");
    for(int iter = 0; iter<max_iter; iter++){
      printf("Iteration -> %d\n",iter);
      objectiveValue = assignCluster(centroids, idx, isolated);
      objectiveValue_difference = objectiveValue - previous_objectiveValue;
      printf("iter: %d, obj=%g, diff = %g obj*tol=%g\n", iter, objectiveValue, objectiveValue_difference, objectiveValue*TOL);
      if (objectiveValue_difference < objectiveValue*TOL)
        break;
      previous_objectiveValue = objectiveValue;
      updateCentroids(centroids, idx, isolated);
    }
}

vector< map<int, double> >  kmeans::initializeCentroids(){
    vector< map<int, double> > centroids;
    for( int i=0;i<k_cluster;i++){
      map<int,double> centroid = sparseMatrix->randomlyPickAnInstance();
      centroids.push_back(centroid);
    }
    return centroids;
}
#endif
