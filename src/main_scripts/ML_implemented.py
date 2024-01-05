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

    def train_test_split_feature_select(self, df: pd.DataFrame, train_size=0.1, r_state=42, feature_select='SelectKBest', n_feature=10, normalizzation=True):
        x = df.drop(self.ml_data.Class_vector, axis=1)
        y = self.ml_data.creating_bina_class(df, self.ml_data.Class_vector, value=2.48)
        X_test, y_test, X_train, y_train = self.ml_data.test_data(x, y, train_size=train_size, r_state=r_state)
        if normalizzation is True:
            X_test, X_train = X_test.apply(self.ml_data.Pareto_Scaling), X_train.apply(self.ml_data.Pareto_Scaling)

        if feature_select == 'SelectKBest':
            selectKbest = SelectKBest(score_func=f_classif, k=n_feature)  # feature selection with prevented data leekage
            selectKbest.fit(X_train, y_train)
            selected_features = selectKbest.get_feature_names_out()
            # print(selected_features)
            return X_test[selected_features], y_test, X_train[selected_features], y_train
        elif feature_select == 'None':
            return X_test, y_test, X_train, y_train
        else:
            return KeyError

    def predictions(self, X_test, y_test, X_train, y_train):
        est = self.est(test_X=X_test, test_y=y_test, train_X=X_train, train_y=y_train)
        forest = est.Forest(n_est=1000)
        svm = est.SVM(kernel='linear')
        bayes = est.Bayes()
        decisoTree = est.DecisionTree()
        quadra = est.QuadraticDiscrAnal()
        MLP = est.MultiPerceptron(layers_sizes=(100,50), activ='identity', max_i=600)
        matrix = pd.concat([est.confusion_matrix(forest, 'Forest'),
                            est.confusion_matrix(svm, 'SVM'),
                            est.confusion_matrix(bayes, 'Bayes'),
                            est.confusion_matrix(decisoTree, 'DecisionTree'),
                            est.confusion_matrix(quadra, 'QuadraticDiscriminantAnalasis'),
                            est.confusion_matrix(MLP, 'MultiLayeredPerceptron')
                            ], axis=1)
        return matrix


    def cross_validations(self, df: pd.DataFrame, n_splits, r_state, normalization=True):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=r_state)
        X = df.drop(self.ml_data.Class_vector, axis=1)
        y = df[self.ml_data.Class_vector]
        # y = self.ml_data.creating_bina_class(df, self.ml_data.Class_vector, value=2.48)
        if normalization is True:
            X = X.apply(self.ml_data.Pareto_Scaling)
        for train, test in skf.split(X=X, y=y):
            yield X.iloc[train, :], y[train], X.iloc[test, :], y[test]

    def predictions_crossval(self, cross_val):
        macierze = []
        for train_test in list(cross_val):
            Xtest, ytest, Xtrain, ytrain = train_test
            est = self.est(test_X=Xtest, test_y=ytest, train_X=Xtrain, train_y=ytrain)
            forest = est.Forest(n_est=1000)
            svm = est.SVM(kernel='linear')
            bayes = est.Bayes()
            decisoTree = est.DecisionTree()
            quadra = est.QuadraticDiscrAnal()
            MLP = est.MultiPerceptron(layers_sizes=(100,50), activ='identity', max_i=600)
            matrix = pd.concat([est.confusion_matrix(forest, 'Forest'),
                                est.confusion_matrix(svm, 'SVM'),
                                est.confusion_matrix(bayes, 'Bayes'),
                                est.confusion_matrix(decisoTree, 'DecisionTree'),
                                est.confusion_matrix(quadra, 'QuadraticDiscriminantAnalasis'),
                                est.confusion_matrix(MLP, 'MultiLayeredPerceptron'),
                                ], axis=1)
            macierze.append(matrix.values)
        cross_vali_stats = np.stack(macierze, axis=2)
        mean_cros_stats = np.mean(cross_vali_stats, axis=2)
        resulted_matrix = pd.DataFrame(data=mean_cros_stats, columns=matrix.columns, index=matrix.index)
        return resulted_matrix


    def test_various_SVM(self,X_test, y_test, X_train, y_train):
        est = self.est(test_X=X_test, test_y=y_test, train_X=X_train, train_y=y_train)
        svm1 = est.SVM(kernel='linear')
        svm2 = est.SVM(kernel='poly', degree=3)
        svm3 = est.SVM(kernel='rbf')
        svm4 = est.SVM(kernel='sigmoid')
        matrix = pd.concat([
            est.confusion_matrix(svm1, 'SVM_linear'), #<best one
            est.confusion_matrix(svm2, 'SVM_poly'),
            est.confusion_matrix(svm3, 'SVM_rbf'),
            est.confusion_matrix(svm4, 'SVM_sigmoid'),
        ],
            axis=1)
        return matrix

    def test_varoius_MLP(self, X_test, y_test, X_train, y_train):
        est = self.est(test_X=X_test, test_y=y_test, train_X=X_train, train_y=y_train)
        mlp1 = est.MultiPerceptron(layers_sizes=(10,5), activ='relu', max_i=600)
        mlp2 = est.MultiPerceptron(layers_sizes=(10,5), activ='tanh', max_i=600)
        mlp3 = est.MultiPerceptron(layers_sizes=(10,5), activ='identity', max_i=600)
        mlp4 = est.MultiPerceptron(layers_sizes=(10,5), activ='logistic', max_i=600)
        matrix = pd.concat(
            [
                est.confusion_matrix(mlp1, 'MLP_relu'),
                est.confusion_matrix(mlp2, 'MLP_tanh'),
                est.confusion_matrix(mlp3, 'MLP_identity'),
                est.confusion_matrix(mlp4, 'MLP_logistic'),
            ],
            axis=1
        )
        return matrix