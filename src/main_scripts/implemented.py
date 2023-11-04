import pandas as pd

from src.data_preparation.data_preparation import DataPreparation
from src.machine_learning.estimators import Estimators

class Implementations:
    def __init__(self):
        self.prep = DataPreparation()
        self.est = Estimators

    def dataframe_preparation(self, df: pd.DataFrame, new_columns: list[str], count_amins: list[list[str]]):
        pass

    def clasterization_results(self, df, size=(40,35), font_scale=1.5, fmt='.1f', annota=7):
        for indx in range(len(self.prep.columns)):
            claster = self.prep.clsterization(indx, df)
            auto_corr = lambda x: (x, self.prep.correlation(claster[x]))
            corrls = dict(map(auto_corr, claster.keys()))
            for k in corrls.keys():
                self.prep.heatmaps(corrls[k], title=k, size=size, font_scale=font_scale, fmt=fmt, annota=annota)


    def predictions(self, df, y_col, pre_value):
        masked_df = self.prep.mask_nans(df)
        xtest, ytest, xtrain, ytrain = self.prep.test_data(masked_df[self.prep.cechy],
                                                           masked_df[y_col],
                                                           train_size=0.8,
                                                           r_state=10)
        est = self.est(xtest, ytest, xtrain, ytrain)
        SVM_pred = est.SVM('rbf', 4)
        bayes_pred = est.Bayes()
        tree = est.DecisionTree()
        forest = est.Forest(n_est=1000)
        estimation_df = pd.concat([est.confusion_matrix(SVM_pred, pre_value, 'SVM'),
                                   est.confusion_matrix(bayes_pred, pre_value, "Bayes"),
                                   est.confusion_matrix(tree, pre_value, 'DecisionTree'),
                                   est.confusion_matrix(forest, pre_value, 'RandomForest')], axis=1)
        return estimation_df


    def predictions_one_hot_encoding(self, df):
        pass

    def predictions_cross_validations(self, df):
        pass