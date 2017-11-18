import math
import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix
class kmeans():
    def __init__(self, k_cluster = 5, max_iter = 100, saveFile = True, outputFile = 'SDE.mat'):
        # :param sparseMatrix (Compressed Sparse Column)-> sparse_matrix
        # :param k_cluster (int) -> The number of cluster
        # :param max_iter (int) -> The maximum of max_tier times the algorithm terminates
        self.k_cluster = k_cluster
        self.sparseMatrix = None
        self.max_iter = max_iter
        self.saveFile = saveFile
        self.outputFile = outputFile
    def initializeCentroids(self):
        # initialize all centroids
        centroids = list()
        # repeat k_cluster times
        for k in range(self.k_cluster):
            # random pick a random instance as the centroid
            centroid = self.sparseMatrix.pickRandomInstance()
            # put it in the list, centroids
            centroids.append(centroid)
        # return the set of these randomly generated centroids
        return centroids
    def normalizeCentroids(self,centroids):
        for centroid in centroids:
            length = 0
            for featureIndex, value in centroid.items():
                length += value **2
            if length>0:
                length = 1/math.sqrt(length)
                for featureIndex, value in centroid.items():
                    centroid[featureIndex] = value * length
    def AssignCluster(self,centroids,idx,isolated):
        # Reset {max_similarity}
        max_similarity = np.zeros(self.sparseMatrix.getInstanceSize())
        for i in range(self.k_cluster):
            centroid = centroids[i]
            #relevant instance set
            relevantInstanceSet = self.sparseMatrix.getRelevantInstanceSetByFeatureIndexAndCentroid(centroid)
            #print relevantInstanceSet
            for instanceIndex in relevantInstanceSet:
                similarity = self.sparseMatrix.calculateSimilarity(instanceIndex, centroid)
                if similarity > max_similarity[instanceIndex]:
                    # replace with the max_similarity
                    max_similarity[instanceIndex] = similarity
                    # assign the instance to k cluster
                    idx[instanceIndex] = i
                    #print 'instance:',instanceIndex,'belongs to ',idx[instanceIndex]
        while isolated:
            isolated.pop()
        obj = 0.0
        for instanceIndex in range(self.sparseMatrix.getInstanceSize()):
            obj += max_similarity[instanceIndex]
            if max_similarity[instanceIndex] < 1e-9:
              isolated.append(instanceIndex)
        return obj
    def updateCentroid(self,instanceIndex,centroid):
        '''
             search the given centroid for the existence of a feature
             and increment by one

             +---------------------------+
             | Edge  |  Features         |
             |---------------------------|
             | (x,y) | 1 2 3 4 5 6 7 8 9 |
             |---------------------------|
             | (1,3) | 1 0 1 0 0 0 0 0 0 |
             | (2,4) | 0 1 0 1 0 0 0 0 0 |
             | (1,8) | 1 0 0 0 0 0 0 1 0 |
             +---------------------------+

             instanceIndex corresponds to the row number of "Edge"
             featureIndex corresponds to the column numbers of "Feature"
             featureIndex_1 corresponds to x value
             featureIndex_2 corresponds to y value

             centroid is a map variable which stores key->value pairs with
             keys in a sorted order for quick searching

             increment the feature indices (x, y) value by one if it is found
             otherwise set it to one...
        '''
        for featureIndex in  self.sparseMatrix.InstanceToFeature[instanceIndex]:
            if featureIndex in centroid:
                centroid[featureIndex] += 1
            else:
                centroid[featureIndex] = 1

    def updateCentroids(self,centroids,idx,isolated):
        numCluster = np.zeros(self.k_cluster, dtype=np.int)
        for centroid in centroids:
            centroid.clear()
        for instanceIndex in range(self.sparseMatrix.getInstanceSize()):
            numCluster[idx[instanceIndex]] += 1
            self.updateCentroid(instanceIndex, centroids[idx[instanceIndex]])
            #print 'instance:',instanceIndex,'belongs to ',idx[instanceIndex]
        for i in range(self.k_cluster):
            #print 'Number of cluster[%d] = %d ' %( i,numCluster[i])
            # if any centroid doesn't have any instance, just randomly assign a instance to it
            if numCluster[i] == 0:
                centroids[i] = self.sparseMatrix.pickInstanceFromIsolated(isolated)
        self.normalizeCentroids(centroids)
    def fit(self,sparseMatrix):
        # :param sparseMatrix (Compressed Sparse Column)-> sparse_matrix
        self.sparseMatrix = sparseMatrix

        # initialize the centroid of cluster
        centroids = self.initializeCentroids()
        # normalize centroids
        self.normalizeCentroids(centroids)
        previous_objectiveValue = 0
        objectiveValue_difference = 0
        tolerance = 0.001
        isolated = []
        idx = np.empty(self.sparseMatrix.getInstanceSize(), dtype=np.int)
        for i in range(self.max_iter):
            print ('Iteration %d' % i)
            # Reset {idx}
            idx.fill(0)
            # Step 1 AssignCluster
            #print 'Step 1 AssignCluster'
            objectiveValue = self.AssignCluster(centroids,idx,isolated)
            objectiveValue_difference = objectiveValue - previous_objectiveValue
            print ('objectiveValue',objectiveValue)
            print ('objectiveValue_difference',objectiveValue_difference)
            if objectiveValue_difference <= objectiveValue * tolerance:
                break
            previous_objectiveValue = objectiveValue
            # Step 2 updateCentroids
            #print 'Step 2 updateCentroids'
            self.updateCentroids(centroids,idx,isolated)
            print ('----------------------------------')

        # save to spraseMatrix
        row = []
        col = []
        data = []
        for i in range(self.k_cluster):
            centroid = centroids[i]
            for key in centroid:
                row.append(key)
                col.append(i)
                data.append(centroid[key])
        self.__cluster_centers = csr_matrix((data, (row, col)))
        if self.saveFile:
            sio.savemat(self.outputFile,{'Extraction':self.__cluster_centers})
        return self

    @property
    def cluster_centers(self):
        return self.__cluster_centers


if __name__ == '__main__':
    from SparseMatrix import SparseMatrix
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", help="k_cluster", type = int)
    parser.add_argument("-f", help="filename")
    parser.add_argument("-o", help="output")
    args = parser.parse_args()
    fileName = args.f
    k_cluster = args.k
    output = args.o
    if fileName is None:
        raise ValueError("main: empty filename")
    elif k_cluster == None or k_cluster <=0:
        raise ValueError("main: invalid number k_cluster")
    else:
        st = time.time()
        s = SparseMatrix(fileName = fileName)
        k = kmeans(k_cluster= k_cluster, saveFile = True, outputFile = output).fit(s)
        print ('Time used:',str(time.time()-st))
