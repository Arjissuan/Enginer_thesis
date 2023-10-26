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

    def efficiency(self, prediction):
        accuracy = np.sum(self.test_class == prediction)/len(self.test_class)
        return accuracy