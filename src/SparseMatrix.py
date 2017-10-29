import random
import scipy.io as sio
import numpy as np
from scipy.sparse import csr_matrix

class SparseMatrix():
    def __init__(self, fileName = 'blogcatalog.mat'):
        self._data = None
        self.edge_size = 0
        self.node_size = 0
        self.load_from_mat(fileName)
    def load_from_mat(self,data):
        # construct a mapping from features to instances
        mat_contents = sio.loadmat(data)
        network = mat_contents['network']
        self.feature_size = network.shape[0]
        self.instance_size = network.nnz / 2
        rows,columns = network.nonzero()
        network_size = network.nnz
        sparse_row = np.zeros(network_size)
        sparse_column = np.zeros(network_size)
        sparse_data = np.ones(network_size)
        count_element = 0
        count_row = 0
        for i in range(network_size):
            node_1 = rows[i]
            node_2 = columns[i]
            if node_2 > node_1:
                sparse_row[count_element] = count_row
                sparse_column[count_element] = node_1
                sparse_row[count_element+1] = count_row
                sparse_column[count_element+1] = node_2
                count_element = count_element + 2
                count_row = count_row +1
        # covert it into sparse matrix
        self._data = csr_matrix((sparse_data, (sparse_row, sparse_column)))
    def getRelevantInstanceSetByFeatureIndexAndCentroid(self,centroid):
        relevantInstanceSet = set()
        # centroid contains featureIndex and value
        for featureIndex in centroid:
            instanceSet, columns = self._data[:,featureIndex].nonzero()
            for instanceIndex in instanceSet:
                relevantInstanceSet.add(instanceIndex)
        return relevantInstanceSet
    def pickRandomInstance(self):
        instanceIndex = random.randint(0, self.getInstanceSize() - 1)
        instance = self.dataToInstance(instanceIndex)
        return instance
    def pickInstanceFromIsolated(self, isolated):
        instanceIndex = 0
        if isolated:
            instanceIndex = isolated.pop()
            return self.dataToInstance(instanceIndex);
        else:
            return self.pickRandomInstance()
    def dataToInstance(self,instanceIndex):
        instance = {}
        rows, featureSet = self._data[instanceIndex,:].nonzero()
        for featureIndex in featureSet:
            value = self._data[instanceIndex,featureIndex]
            instance[featureIndex] = value
        return instance
    def calculateSimilarity(self, instanceIndex, centroid):
        similarity = 0
        rows, featureSet = self._data[instanceIndex,:].nonzero()
        for featureIndex in featureSet:
            if featureIndex in centroid:
                value = self._data[instanceIndex,featureIndex]
                similarity += value * centroid[featureIndex]
        return similarity
    def getFeatureSize(self):
        return self.feature_size
    def getInstanceSize(self):
        return self.instance_size
    def getMatrix(self):
        return self._data
