import math
import numpy as np
class kmeans():
    def __init__(self, k_cluster = 5, max_iter = 100):
        # :param sparseMatrix (Compressed Sparse Column)-> sparse_matrix
        # :param k_cluster (int) -> The number of cluster
        # :param max_iter (int) -> The maximum of max_tier times the algorithm terminates
        self.k_cluster = k_cluster
        self.sparseMatrix = None
        self.max_iter = max_iter
    def fit(self,sparseMatrix):
        # :param sparseMatrix (Compressed Sparse Column)-> sparse_matrix
        self.sparseMatrix = sparseMatrix
        return self
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
            print 'Number of cluster[%d] = %d ' %( i,numCluster[i])
            # if any centroid doesn't have any instance, just randomly assign a instance to it
            if numCluster[i] == 0:
                centroids[i] = self.sparseMatrix.pickInstanceFromIsolated(isolated)
        self.normalizeCentroids(centroids)
    def solve(self):
        # initialize the centroid of cluster
        centroids = self.initializeCentroids()
        previous_objectiveValue = 0
        objectiveValue_difference = 0
        tolerance = 0.001
        isolated = []
        idx = np.empty(self.sparseMatrix.getInstanceSize(), dtype=np.int)
        # normalize centroids
        self.normalizeCentroids(centroids)
        for i in range(self.max_iter):
            print 'Iteration %d' % i
            # Reset {idx}
            idx.fill(0)
            # Step 1 AssignCluster
            print 'Step 1 AssignCluster'
            objectiveValue = self.AssignCluster(centroids,idx,isolated)
            objectiveValue_difference = objectiveValue - previous_objectiveValue
            print 'objectiveValue',objectiveValue
            print 'objectiveValue_difference',objectiveValue_difference
            if objectiveValue_difference <= objectiveValue * tolerance:
                break
            previous_objectiveValue = objectiveValue
            # Step 2 updateCentroids
            print 'Step 2 updateCentroids'
            self.updateCentroids(centroids,idx,isolated)
            print '----------------------------------'
