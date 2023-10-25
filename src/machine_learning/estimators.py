from sklearn import svm


class Estimators:
    def __init__(self, t_c, t_v, l_c, l_v):
        self.test_class = t_c
        self.test_values = t_v
        self.learn_class = l_c
        self.learn_values = l_v

    def SVM(self, kernel):
        clf = svm.SVC(kernel=kernel)
        clf.fit(X=self.learn_values, y=self.learn_class)
        return clf.predict(X=self.test_values)

    def Bayes(self):
        pass

    def efficiency(self, df):
        pass
