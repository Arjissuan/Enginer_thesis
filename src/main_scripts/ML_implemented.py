from src.machine_learning.estimators import Estimators
from src.machine_learning.ML_data import ML_data
from src.main_scripts.implemented import Implementations
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
class ML_implemented:
    def __init__(self, Class_column):
        self.analasis = Implementations(Class_column)
        self.ml_data = ML_data(Class_column)
        self.est = Estimators

    def train_test_split_feature_select(self, df: pd.DataFrame, train_size=0.1, r_state=42, feature_select='SelectKBest', n_feature=20, normalizzation=True):
        x = df.drop(self.ml_data.Class_vector, axis=1)
        y = self.ml_data.creating_bina_class(df, self.ml_data.Class_vector, value=2.48)
        X_test, y_test, X_train, y_train = self.ml_data.test_data(x, y, train_size=train_size, r_state=r_state)
        if normalizzation is True:
            X_test, X_train = X_test.apply(self.ml_data.Pareto_Scaling), X_train.apply(self.ml_data.Pareto_Scaling)

        if feature_select == 'SelectKBest':
            selectKbest = SelectKBest(score_func=f_classif, k=n_feature)  # feature selection without data leekage
            selectKbest.fit(X_train, y_train)
            selected_features = selectKbest.get_feature_names_out()
            # print(selected_features)
            return X_test[selected_features], y_test, X_train[selected_features], y_train
        elif feature_select == 'By_correlation_changes':
            corrmatrix = self.analasis.corelations_presentation()
            self.analasis.corelations_histograms(corrmatrix[3])
            selected_features = input('Choosen: ')
            return X_test[selected_features], y_test, X_train[selected_features], y_train
        else:
            return X_test, y_test, X_train, y_train



    def predictions(self, X_test, y_test, X_train, y_train):
        est = self.est(test_X=X_test, test_y=y_test, train_X=X_train, train_y=y_train)
        forest = est.Forest(n_est=100)
        svm = est.SVM(kernel='rbf', degree=5)
        bayes = est.Bayes()
        decisoTree = est.DecisionTree()
        quadra = est.QuadraticDiscrAnal()
        matrix = pd.concat([est.confusion_matrix(forest, 'Forest'),
                            est.confusion_matrix(svm, 'SVM'),
                            est.confusion_matrix(bayes, 'Bayes'),
                            est.confusion_matrix(decisoTree, 'DecisionTree'),
                            est.confusion_matrix(quadra, 'QuadraticDiscriminantAnalasis')
                            ], axis=1)
        return matrix

    def cross_validations(self, df: pd.DataFrame, n_splits, r_state, normalization=True):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=r_state)
        X = df.drop(self.ml_data.Class_vector, axis=1)
        y = self.ml_data.creating_bina_class(df, self.ml_data.Class_vector)
        if normalization is True:
            X = X.apply(self.ml_data.Pareto_Scaling)
        for train, test in skf.split(X=X, y=y):
            yield X.iloc[train, :], y[train], X.iloc[test, :], y[test]

    def predictions_crossval(self, cross_val):
        macierze = []
        for train_test in list(cross_val):
            Xtest, ytest, Xtrain, ytrain = train_test
            est = self.est(test_X=Xtest, test_y=ytest, train_X=Xtrain, train_y=ytrain)
            forest = est.Forest(n_est=100)
            svm = est.SVM(kernel='rbf', degree=5)
            bayes = est.Bayes()
            decisoTree = est.DecisionTree()
            quadra = est.QuadraticDiscrAnal()
            matrix = pd.concat([est.confusion_matrix(forest, 'Forest'),
                                est.confusion_matrix(svm, 'SVM'),
                                est.confusion_matrix(bayes, 'Bayes'),
                                est.confusion_matrix(decisoTree, 'DecisionTree'),
                                est.confusion_matrix(quadra, 'QuadraticDiscriminantAnalasis')
                                ], axis=1)
            macierze.append(matrix.values)
        cross_vali_stats = np.stack(macierze, axis=2)
        mean_cros_stats = np.mean(cross_vali_stats, axis=2)
        resulted_matrix = pd.DataFrame(data=mean_cros_stats, columns=matrix.columns, index=matrix.index)
        return resulted_matrix