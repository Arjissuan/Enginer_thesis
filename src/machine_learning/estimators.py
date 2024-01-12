import pandas as pd
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import numpy as np


class Estimators:
    def __init__(self, test_X, test_y, train_X, train_y):
        self.test_values = test_X
        self.test_class = test_y
        self.train_values = train_X
        self.train_class = train_y

    def SVM(self, kernel, degree=5):
        clf = svm.SVC(kernel=kernel, degree=degree, )
        clf.fit(X=self.train_values, y=self.train_class)
        return clf.predict(X=self.test_values)

    def Bayes(self):
        gnb = GaussianNB()
        gnb.fit(X=self.train_values, y=self.train_class)
        return gnb.predict(X=self.test_values)

    def DecisionTree(self):
        wood = tree.DecisionTreeClassifier(random_state=42)
        wood.fit(X=self.train_values, y=self.train_class)
        return wood.predict(X=self.test_values)

    def Forest(self, n_est=100):
        forest = RandomForestClassifier(n_estimators=n_est, random_state=42)
        forest.fit(X=self.train_values, y=self.train_class)
        return forest.predict(X=self.test_values)

    def MultiPerceptron(self, layers_sizes=(5,3,1), activ='tanh', max_i=400):
        mp = MLPClassifier(hidden_layer_sizes=layers_sizes, activation=activ, max_iter=max_i, alpha=0.5)
        mp.fit(X=self.train_values, y=self.train_class)
        return mp.predict(X=self.test_values)

    def confusion_matrix(self, prediction, estimator):
        TP = np.sum(np.logical_and(self.test_class, prediction))
        TN = np.sum(np.logical_and(np.logical_not(self.test_class), np.logical_not(prediction)))
        FP = np.sum(np.logical_and(np.logical_not(self.test_class), prediction))
        FN = np.sum(np.logical_and(self.test_class, np.logical_not(prediction)))

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