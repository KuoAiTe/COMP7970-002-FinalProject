#include "par_kmeans.h"
#include <cmath>

#ifndef K_MEANS
#define K_MEANS

using namespace std;

kmeans::kmeans(int _k,int _max){
    
    k_cluster = _k;  // number of centroids
    max_iter = _max; // max iterations to normalize
    
}

void kmeans::fit(SparseMatrix* _sparseMatrix){
    
    sparseMatrix = _sparseMatrix; // assigns data to class instance
    
}

void kmeans::updateCentroids(vector< map<int,double> > &centroids, int *idx, stack<int> &isolated) {
    
    // this function recalculates the values of each centroid
    // based on the assigned data elements
    
    int *numCluster = new int[k_cluster];
    memset(numCluster, 0, sizeof(int) * k_cluster);
    
    for (int i=0; i<centroids.size(); i++) {
        centroids[i].clear();
    }
    
    for (int i=0; i < sparseMatrix->getRankInstanceSize(); i++) {
        numCluster[idx[i]]++;
        sparseMatrix->updateCentroid(i, centroids[idx[i]]);
    }
    
    for (int i=0; i<centroids.size(); i++) {

    #ifdef DEBUG
    #if DEBUG > 6
        dbprintf("numCluster[%d] = %d\n", i, numCluster[i]);
    #endif
    #endif
    
    if (numCluster[i] == 0 ) {
            centroids[i] = sparseMatrix->pickInstanceFromIsolated(isolated);
        }
    }
    
    normalizeCentroids(centroids);
    
}

void kmeans::normalizeCentroids(vector< map<int,double> > &centroids){
    
    for (int i=0; i<centroids.size(); i++){
        
        map<int,double> &instance = centroids[i];
        double length = 0;
        
        // centroid[0] = instance = { instance[1] { 1 }, instance[6] { 1 } }
        
        // iterator->first = key, iterator->second = similarity measure
        
        for (map<int,double>::iterator it = instance.begin(); it!=instance.end(); it++) {
            
            length += it->second * it->second;
            
            #ifdef DEBUG
            #if DEBUG > 6
                dbprintf("length += %f * %f = %f\n", it->second, it->second, length);
            #endif
            #endif
            
        }
        
        if (length > 0) {
            
            length = 1/sqrt(length);
            
            for (map<int,double>::iterator it = instance.begin(); it!=instance.end(); it++) {
                
                instance[it->first] = it->second * length;
                
                #ifdef DEBUG
                #if DEBUG > 6
                    dbprintf("normalized length = %f instance[it->first] = %f * %f = %f\n", length, it->second, length, instance[it->first]);
                #endif
                #endif
                
            }
            
        }
        
    }
    
}

double kmeans::assignCluster(vector< map<int,double> > &centroids, int * idx, stack<int> &isolated) {
    
    set<int>::iterator it;
    
    int instanceIndex;
    int instanceSize = sparseMatrix->getInstanceSize();
    
    double * maxSimilarity = new double[instanceSize];
    double Similarity;
    
    memset(idx, 0, sizeof(int) * instanceSize);
    memset(maxSimilarity, 0, sizeof(double)*instanceSize);
    
    int *RelevantInstanceSet = (int *)malloc(sizeof(int)*instanceSize);
    
    dbprintf("assigning set of size %d\n", instanceSize);
    
    for (int i=0; i<centroids.size(); i++){
        
        int count = sparseMatrix->getRelevantInstanceSetByFeatureIndexAndCentroid(centroids[i], RelevantInstanceSet);
        
        dbprintf("got %d sets of relevant instances\n", count);
        
        // now we have a list of integers corresponding to edge records in the data
        // iterate through it and calculate similarity...
        
        for(int j=0; j<count; j++) {
            
            instanceIndex = RelevantInstanceSet[j];
            
            Similarity = sparseMatrix->calculateSimilarity(instanceIndex, centroids[i]);
            
            // assign the largest index found so far
            
            if (Similarity > maxSimilarity[instanceIndex]){
                
                dbprintf("new similarity for instanceIndex = %d\n", instanceIndex);
                maxSimilarity[instanceIndex] = Similarity;
                
                idx[instanceIndex] = i;
                
            }
            
        }
        
    }
    
    while(!isolated.empty()) {
        isolated.pop();
    }
    
    double obj = 0;
    
    for (int i=0; i<instanceSize; i++){
        obj += maxSimilarity[i];
        if (maxSimilarity[i] < 1e-9) {
            isolated.push(i);
        }
    }
    
    return obj;
    
}

void kmeans::solve(int nprocs, int rank) {
    
    dbprintf("Solving...\n");
    
    vector< map<int, double> > centroids = initializeCentroids();
    
    dbprintf("normalizing centroids...\n");
    
    normalizeCentroids(centroids);
    
    dbprintf("initializing variables...\n");
    
    int instanceSize = sparseMatrix->getRankInstanceSize();
    int * idx = new int[instanceSize];
    double previous_objectiveValue=0, objectiveValue, objectiveValue_difference,TOL = 0.001;
    stack<int> isolated = stack<int>();
    
    dbprintf("Start Clustering\n");
    
    for(int iter = 0; iter<max_iter; iter++) {
        
        dbprintf("Iteration -> %d\n",iter);
        
        objectiveValue = assignCluster(centroids, idx, isolated);
        
        objectiveValue_difference = objectiveValue - previous_objectiveValue;
        
        dbprintf("iter: %d, obj=%g, diff = %g obj*tol=%g\n", iter, objectiveValue, objectiveValue_difference, objectiveValue*TOL);
        
        if (objectiveValue_difference < objectiveValue*TOL) { break; }
        
        previous_objectiveValue = objectiveValue;
        
        updateCentroids(centroids, idx, isolated);
        
    }
    
}

vector< map<int, double> > kmeans::initializeCentroids() {
    
    // will return k randomly selected centroid vectors
    // each as a map of only two index -> value elements, i.e...
    //
    // centroid[0] =  { instance[1] = { 1 }, instance[6] = { 1 } }
    //
    // instance = map<int, double>
    //
    // the second element in each centroid instance is a type double
    // since the centroids are normalized in the 0 - 1 range...
    //
    // instance[0] = 0.0000
    //
    
    vector< map<int, double> > centroids;
    
    for(int i=0; i<k_cluster; i++){
        
        // get a random data element in map<int,double> type...
        
        map<int,double> centroid = sparseMatrix->randomlyPickAnInstance();
        
        #ifdef DEBUG
            map<int,double>::iterator pos;
            for (pos = centroid.begin(); pos != centroid.end(); ++pos) {
                dbprintf("centroid[%d]: %f, key: %d, value: %f\n", i, centroid[i], pos->first, pos->second);
            }
        #endif
        
        // add the new centroid[i] = { { 22, 1.000 }, {42, 1.000 } } to set ...
        
        centroids.push_back(centroid);
        
    }
    
    return centroids;
    
}

#endif
