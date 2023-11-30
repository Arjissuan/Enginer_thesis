from src.machine_learning.estimators import Estimators
from src.machine_learning.ML_data import ML_data
import pandas as pd
import numpy as np
class ML_implemented:
    def __init__(self, Class_column):
        self.ml_data = ML_data(Class_column)
        self.est = Estimators
    def predictions(self, df, train_size=0.3, r_state=42):
        x = df.drop(self.ml_data.Class_vector, axis=1)
        y = self.ml_data.creating_bina_class(df, self.ml_data.Class_vector, value=2.48)
        X_test, y_test, X_train, y_train = self.ml_data.test_data(x,y,train_size=train_size, r_state=r_state)
        est = self.est(test_v=X_test, test_c=y_test, train_v=X_train, train_c=y_train)
        forest = est.Forest(50)
        svm = est.SVM('rbf', 3)
        bayes = est.Bayes()
        decisoTree = est.DecisionTree()
        line = est.Linear()
        matrix = pd.concat([est.confusion_matrix(forest, 'Forest'),
                            est.confusion_matrix(svm, 'SVM'),
                            est.confusion_matrix(bayes, 'Bayes'),
                            est.confusion_matrix(decisoTree, 'DecisionTree'),
                            est.confusion_matrix(line, 'Linear')], axis=1)
        return matrix

    def predictions_one_hot_encoding(self, df):
        pass

    def predictions_cross_validations(self, df):
        pass