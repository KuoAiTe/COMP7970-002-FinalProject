import random
import scipy.io as sio
import numpy as np
import math
import scipy
from Accuracy import Accuracy
from sklearn.svm import LinearSVC
from scipy.sparse import csr_matrix

class linearSVM:
    def __init__(self,dataFile= './data/test.mat',extractionFile = './real.mat',portion = 0.9):
        self.group = sio.loadmat(dataFile)["group"]
        self.extraction = sio.loadmat(extractionFile)["Extraction"]
        # Do not change. This is what the author setup initially.
        self.tol = 0.01
        self.C = 500
        self.dual = False
    def run(self):
        group_data_size, group_categories_size = self.group.shape
        # HoldOut Cross-Validation
        random_indices = np.arange(group_data_size)
        #np.random.shuffle(random_indices)

        # Set the Portion
        portion = 0.9
        boundary = math.ceil(group_data_size*portion)

        train_indices = random_indices[:boundary]
        test_indices = random_indices[boundary:]
        test_size = len(test_indices)

        trainingSet = self.extraction[train_indices]
        testSet = self.extraction[test_indices]

        # trainingSet_Label : group_data_size * group_categories_size
        # testSet_Label : group_data_size * group_categories_size
        trainingSet_Label = self.group[train_indices]
        testSet_Label = self.group[test_indices].todense()
        predictionScore = np.zeros((test_size,group_categories_size))
        prediction = np.zeros((test_size,group_categories_size))

        for i in range(group_categories_size):
            trainingLabel = np.ravel(trainingSet_Label[:,i].todense())
            for j in range(len(trainingLabel)):
                if trainingLabel[j] == 0: trainingLabel[j]=-1
            clf = LinearSVC( C= self.C, dual=self.dual, tol = self.tol)
            clf.fit(trainingSet, trainingLabel)

            p = clf.decision_function(testSet)
            for j in range(len(p)):
                predictionScore[j][i] = p[j]

        row=[]
        col=[]
        data=[]
        instance_correct = np.zeros(test_size)
        for i in range(test_size):
            s = sorted(range(len(predictionScore[i,:])),key = predictionScore[i,:].__getitem__, reverse= True)
            numLabel = len(testSet_Label[i].nonzero()[0])
            counter = numLabel
            for j in range(numLabel):
                row.append(i)
                col.append(s[j])
                data.append(1)
                expectedLabel = testSet_Label[i , s[j]]
                predictLabel = prediction[i , s[j]] = 1
                if predictLabel == expectedLabel:
                    counter -= 1
            if counter == 0:
                instance_correct[i] = 1
        result = csr_matrix((data, (row, col)))
        #np.logical_xor(prediction,)
        All= Accuracy()
        All._accuracy = len(instance_correct.nonzero()[0]) / test_size
        macro_F1 = np.zeros(group_categories_size)
        macro_recall = np.zeros(group_categories_size)
        macro_precision = np.zeros(group_categories_size)
        a = [Accuracy() for _ in range(group_categories_size)]
        for i in range(group_categories_size):
            p = prediction[:,i]
            expected_labels = testSet_Label[:,i]
            a[i].generateResults(expected_labels,p)
            macro_precision[i] =a[i].precision
            macro_recall[i] =a[i].recall
            macro_F1[i] = a[i].f1
            All.addResults(a[i])
        print(All)
        print('macro_precision',np.mean(macro_precision))
        print('macro_recall',np.mean(macro_recall))
        print('macro_F1',np.mean(macro_F1))

if __name__ == '__main__':
    from SparseMatrix import SparseMatrix
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", help="groupFile")
    parser.add_argument("-s", help="SDE File")
    parser.add_argument("-p", help="portion", type = float)
    args = parser.parse_args()
    groupFile = args.g
    SDE_File = args.s
    portion = args.p
    l = linearSVM(dataFile= groupFile,extractionFile = SDE_File,portion =portion)
    l.run()
