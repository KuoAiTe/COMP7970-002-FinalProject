import random
import scipy.io as sio

class SparseMatrix():
    def __init__(self, fileName = 'blogcatalog.mat'):
        self._data = None
        self.edge_size = 0
        self.node_size = 0
        self.InstanceToFeature = list()
        self.FeatureToInstance = list()
        self.load_from_mat(fileName)
    def load_from_mat(self,data):
        # construct a mapping from features to instances
        mat_contents = sio.loadmat(data)
        network = mat_contents['network']
        self.feature_size = network.shape[0]
        self.instance_size = network.nnz // 2
        rows,columns = network.nonzero()
        network_size = network.nnz
        count_element = 0
        count_row = 0
        for i in range(self.instance_size):
            self.InstanceToFeature.append(list())
        for featureIndex in range(self.feature_size):
            self.FeatureToInstance.append(list())
        for i in range(network_size):
            node_1 = rows[i]
            node_2 = columns[i]
            if node_2 > node_1:
                # instnace to feature
                self.InstanceToFeature[count_row].append(node_1)
                self.InstanceToFeature[count_row].append(node_2)
                # feature to instance
                self.FeatureToInstance[node_1].append(count_row)
                self.FeatureToInstance[node_2].append(count_row)
                count_element = count_element + 2
                count_row = count_row +1
    def getRelevantInstanceSetByFeatureIndexAndCentroid(self,centroid):
        relevantInstanceSet = set()
        # centroid contains featureIndex and value
        for featureIndex in centroid:
            for instanceIndex in self.FeatureToInstance[featureIndex]:
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
        for featureIndex in self.InstanceToFeature[instanceIndex]:
            instance[featureIndex] = 1
        return instance
    def calculateSimilarity(self, instanceIndex, centroid):
        similarity = 0
        for featureIndex in self.InstanceToFeature[instanceIndex]:
            if featureIndex in centroid:
                similarity += centroid[featureIndex]
        return similarity
    def getFeatureSize(self):
        return self.feature_size
    def getInstanceSize(self):
        return self.instance_size
    def getMatrix(self):
        return self._data
