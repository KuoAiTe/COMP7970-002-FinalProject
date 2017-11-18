class Accuracy:
    """
     This class is for the result accuracy of any algorithm.
     Additional metrics can be added here while maintaining compatibility with other modules
     """

    def __init__(self):
        # Metrics. T: True; F: False; P: Positive; N: Negative
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self._accuracy = 0
    @property
    def accuracy(self):
        return self._accuracy
    @property
    def precision(self):
        numerator = self.TP
        denumerator = self.TP+self.FP
        if denumerator ==0 : denumerator = 1
        return numerator / denumerator
    @property
    def recall(self):
        numerator = self.TP
        denumerator = self.TP+self.FN
        if denumerator ==0 : denumerator = 1
        return numerator / denumerator
    @property
    def f1(self):
        r = self.recall
        p = self.precision
        numerator = 2*r*p
        denumerator = r+p
        if denumerator ==0 : denumerator = 1
        return numerator / denumerator
    def __str__(self):
        return self.getString()
    def getString(self):
        string = "Accuracy: " + str(self.accuracy) + "\n" + \
                "micro_Precision: " + str(self.precision) + "\n" + \
                "micro_Recall: " + str(self.recall) + "\n" + \
                "micro_F1: " + str(self.f1)
        return string
    def generateResults(self, expected_labels, predicted_labels):
        """
        This function generates a result object with TN, TP, FN, FP and accuracy based on expected and actual values.
        :param expected_labels:
        :param predicted_labels:
        :return: Accuracy
        """
        # The result object to return

        for i in range(len(predicted_labels)):
            if predicted_labels[i] > 0  and expected_labels[i] > 0:
                self.TP += 1
            elif predicted_labels[i] > 0 and expected_labels[i] <= 0:
                self.FP += 1
            elif predicted_labels[i] <= 0 and expected_labels[i] > 0:
                self.FN += 1
            elif predicted_labels[i] <= 0 and expected_labels[i] <= 0:
                # we don't count this
                pass
    def addResults(self, result):
        if not isinstance(result, Accuracy):
            raise ValueError("Accuracy:addResults result needs to be of type Accuracy")
        self.TP += result.TP
        self.FP += result.FP
        self.FN += result.FN
