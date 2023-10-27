import pandas as pd
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np



class Estimators:
    def __init__(self, test_v, test_c, train_v,  train_c):
        self.test_values = test_v
        self.test_class = test_c
        self.train_values = train_v
        self.train_class = train_c

    def SVM(self, kernel, degree):
        clf = svm.SVC(kernel=kernel, degree=degree)
        clf.fit(X=self.train_values, y=self.train_class)
        return clf.predict(X=self.test_values)

    def Bayes(self):
        gnb = GaussianNB()
        gnb.fit(X=self.train_values, y=self.train_class)
        return gnb.predict(X=self.test_values)

    def DecisionTree(self):
        wood = tree.DecisionTreeClassifier()
        wood.fit(X=self.train_values, y=self.train_class)
        return wood.predict(X=self.test_values)

    def Forest(self, n_est):
        forest = RandomForestClassifier(n_estimators=n_est)
        forest.fit(X=self.train_values, y=self.train_class)
        return forest.predict(X=self.test_values)

    def confusion_matrix(self, prediction, predicted_value, estimator):
        actual = self.test_class == predicted_value
        predicted = prediction == predicted_value
        actual_non = self.test_class != predicted_value
        predicted_non = prediction != predicted_value

        TP = np.sum(np.logical_and(actual, predicted))
        TN = np.sum(np.logical_and(actual_non, predicted_non))
        FP = np.sum(np.logical_and(actual_non, predicted))
        FN = np.sum(np.logical_and(actual, predicted_non))

        accuracy = (TN+TP)/(TP+TN+FP+FN)
        error = (FP+FN)/(TP+TN+FP+FN)

        sens = TP/(TP+FN)
        spec = TN/(TN+FP)
        PPV = TP/(TP+FP)
        NPV = TN/(TN+FN)
        matrix = pd.DataFrame(data=[TP,
                                    TN,
                                    FP,
                                    FN,
                                    accuracy,
                                    error,
                                    sens,
                                    spec,
                                    PPV,
                                    NPV],
                              index=['True Positive',
                                       'True Negative',
                                       'False Positive',
                                       'False Negative',
                                       'Accuracy',
                                       'Error',
                                       'Sensitivity',
                                       'Specificity',
                                       'PPV',
                                       'NPV'],
                              columns=[estimator])
        return matrix