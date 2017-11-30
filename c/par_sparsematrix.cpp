#include<vector>
#include <mpi.h>
#include "par_sparsematrix.h"

#ifndef SPARSE_MATRIX
#define SPARSE_MATRIX

using namespace std;

SparseMatrix::SparseMatrix() {
    
}

map<int,double> SparseMatrix::pickInstanceFromIsolated(stack <int> &isolated) {
    
    if (!isolated.empty()){
        
        int idx = isolated.top();
        isolated.pop();
        return getInstanceFromMatrix(idx);
        
    } else {
        
        return randomlyPickAnInstance();
        
    }
    
}

void SparseMatrix::updateCentroid(int instanceIndex, map<int,double> &centroid) {
    
    // search the given centroid for the existence of a feature
    // and increment by one
    
    // +---------------------------+
    // | Edge  |  Features         |
    // |---------------------------|
    // | (x,y) | 1 2 3 4 5 6 7 8 9 |
    // |---------------------------|
    // | (1,3) | 1 0 1 0 0 0 0 0 0 |
    // | (2,4) | 0 1 0 1 0 0 0 0 0 |
    // | (1,8) | 1 0 0 0 0 0 0 1 0 |
    // +---------------------------+
    
    // instanceIndex corresponds to the row number of "Edge"
    // featureIndex corresponds to the column numbers of "Feature"
    // featureIndex_1 corresponds to x value
    // featureIndex_2 corresponds to y value
    
    int featureIndex_1 = getFeaturesByInstance[instanceIndex].first;  // x
    int featureIndex_2 = getFeaturesByInstance[instanceIndex].second; // y
    
    // centroid is a map variable which stores key->value pairs with
    // keys in a sorted order for quick searching
    
    // increment the feature indices (x, y) value by one if it is found
    // otherwise set it to one...
    
    // Q: My understanding so far is that the value of any feature index would be 1.
    //    Why is the value incremented here if already found?
    
    centroid[featureIndex_1] = (centroid.find(featureIndex_1) != centroid.end()) ? centroid[featureIndex_1] + 1 : 1;
    centroid[featureIndex_2] = (centroid.find(featureIndex_2) != centroid.end()) ? centroid[featureIndex_2] + 1 : 1;
    
}

map<int,double> SparseMatrix::randomlyPickAnInstance() {
    
    // will return a feature instance as a collection of two
    // values mapped to a random index between 0 and the
    // size of input
    
    int instanceIndex = rand() % n_size;
    
    map<int,double> instance = getInstanceFromMatrix(instanceIndex);
    
    return instance;
    
}

map<int,double> SparseMatrix::getInstanceFromMatrix(int instanceIndex) {
    
    // will return a feature instance as a collection of two
    // values mapped to an index, i.e...
    //
    // instance[1] = 1.00000
    // instance[6] = 1.00000
    //
    
    map<int,double> instance;
    
    int featureIndex_1 = getFeaturesByInstance[instanceIndex].first;
    int featureIndex_2 = getFeaturesByInstance[instanceIndex].second;
    
    instance[featureIndex_1] = 1;
    instance[featureIndex_2] = 1;
    
    return instance;
    
}

int SparseMatrix::getRelevantInstanceSetByFeatureIndexAndCentroid(map<int,double> &centroid, int *relevantInstanceSet) {
    
    // assigns a list of integers corresponding to the edge record number 0-n
    // returns the count of integers in the list
    // this was changed from a set<int> because that datatype is not thread safe

    int count = 0;
    int featureIndex, instanceIndex;
    
    for(map<int,double>::iterator it = centroid.begin(); it != centroid.end(); it++) {
        
        #if DEBUG > 6
        #ifdef DEBUG
            dbprintf("Finding relevant instances for centroid %d,%f\n", it->first, it->second);
        #endif
        #endif
        
        featureIndex = it->first;
        
        // relevantInstance populated in loadFile as a vector<vector<int>>
        // that maps a given node to its edge number 0-n

        for(int i=0; i < relevantInstance[featureIndex].size(); i++) {
                
            instanceIndex = relevantInstance[featureIndex][i];
            relevantInstanceSet[i] = instanceIndex;
            count++;
                
        }

    }
    
    return count;
    
}

double SparseMatrix::calculateSimilarity(int instanceIndex, map<int,double> &centroid){
    
    // determines if any of the the endpoints of a given edge match the centroid
    // endpoints and assigns a similarity measure
    //
    // note: the centroid key is the feature (node) number and the value is
    // the similarity measure of double type, initially set to 1.0000, i.e....
    //
    // centroid = { { 9 => 1.0000 }, { 12 => 1.0000 } }
    // centroid[9] = 1.0000, centroid[12] = 1.0000
    
    double similarity = 0;
    
    int featureIndex_1 = getFeaturesByInstance[instanceIndex].first;
    int featureIndex_2 = getFeaturesByInstance[instanceIndex].second;
    
    if(centroid.find(featureIndex_1) != centroid.end()) {

    	similarity += centroid[featureIndex_1];

        #ifdef DEBUG
        #if DEBUG > 6
            dbprintf("FOUND getFeaturesByInstance[%d].first = centroid[%d] = %f, similarity = %f\n", instanceIndex, featureIndex_1, centroid[featureIndex_1], similarity);
        #endif
        #endif
    
    }
    
    //similarity += (centroid.find(featureIndex_1) != centroid.end()) ? centroid[featureIndex_1] : 0;
    
    if(centroid.find(featureIndex_2) != centroid.end()) {
        
        similarity += centroid[featureIndex_2];
        
        #ifdef DEBUG
            #if DEBUG > 6
                dbprintf("FOUND getFeaturesByInstance[%d].second = centroid[%d] = %f, similarity = %f\n", instanceIndex, featureIndex_2, centroid[featureIndex_2], similarity);
            #endif
        #endif
   
    }
    
    //similarity += (centroid.find(featureIndex_2) != centroid.end()) ? centroid[featureIndex_2] : 0;
    
    return similarity;
    
}

bool SparseMatrix::loadFile(const char* filename, int rank) {
    
    FILE *fp = fopen(filename, "r");
    
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    
    if (fp) {
        
        int node_1, node_2;
        m_size = n_size = 0;
        getFeaturesByInstance.clear();
        relevantInstance.clear();
        
        // build m -> nodes n -> edges matrix
        // m = nodes, n = edges_size
        
        while (fscanf(fp, "%d,%d", &node_1, &node_2) !=EOF) {
            
            // set nodes size to either the node value or the record
            // number, whichever is largest, i.e...
            // 1,2 >0? m_size=1, >1? m_size=2; n_size=0
            // 1,3 >2? m_size=2, >2? m_size=3; n_size=1
            // 2,5 >3? m_size=3, >3? m_size=5; n_size=2
            // 2,5 >5? m_size=5, >5? m_size=5; n_size=3
            // 3,9 >5? m_size=5, >5? m_size=9; n_size=4
            
            m_size = (node_1 > m_size) ? node_1 : m_size;
            m_size = (node_2 > m_size) ? node_2 : m_size;
            
            // add the "feature" (pair of points) to the features vector
            // the instance index is assigned in order...
            
            getFeaturesByInstance.push_back(make_pair(node_1, node_2));
            
            n_size++;
            
        }
        
        // build feature to instance
        
        for(int i = 0; i < m_size; i++) {
            
            // add empty vectors for each feature...
            
            relevantInstance.push_back(vector<int>());
            
        }
        
        int range = n_size / n_procs;
        int offset = rank * range;
        
        dbprintf("loading input from %d to %d\n", offset, (range + offset - 1));
        
        for(int instanceIndex = offset; instanceIndex < (range + offset - 1); instanceIndex++) {
            
            // assign more granular mapping of feature to instance
            // allows each node to be mapped back to a particular
            // edge (i.e. an index within 0-n edges)
            
            // relevantInstance is vector<vector<int>>
            
            // featureIndex_1 = getFeaturesByInstance[0].first = 1  ; relevantInstance[1][0] = 0
            // featureIndex_2 = getFeaturesByInstance[0].second = 2 ; relevantInstance[2][0] = 0
            // featureIndex_1 = getFeaturesByInstance[1].first = 1  ; relevantInstance[1][1] = 1
            // featureIndex_1 = getFeaturesByInstance[1].second = 3 ; relevantInstance[3][0] = 1
            // featureIndex_1 = getFeaturesByInstance[2].first = 2  ; relevantInstance[2][1] = 2
            // featureIndex_1 = getFeaturesByInstance[2].second = 5 ; relevantInstance[5][0] = 2
            // featureIndex_1 = getFeaturesByInstance[3].first = 2  ; relevantInstance[2][2] = 3
            // featureIndex_1 = getFeaturesByInstance[3].second = 5 ; relevantInstance[5][1] = 3
            // featureIndex_1 = getFeaturesByInstance[4].first = 3  ; relevantInstance[3][1] = 4
            // featureIndex_1 = getFeaturesByInstance[4].second = 9 ; relevantInstance[9][0] = 4
            //
            // relevantInstance[2] = { 0,2,3 }
            //
            
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

int SparseMatrix::getNodeSize() {
    return m_size;
}

int SparseMatrix::getRankInstanceSize() {
    return n_size / n_procs;
}

int SparseMatrix::getInstanceSize() {
    return n_size - 1;
}

#endif

