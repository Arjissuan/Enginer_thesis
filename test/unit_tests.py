from testy import Tests
from src.machine_learning.estimators import Estimators

def clasterization_resutls():
    tt = Tests()
    for idx in range(len(tt.columns)):
        clastered = tt.clsterization(idx)
        auto_corr = lambda x: (x, tt.correlation(clastered[x]))
        correlated = dict(map(auto_corr, clastered.keys()))
        for key in correlated.keys():
            tt.heatmaps(correlated[key], key)


if __name__ == "__main__":
    tt = Tests()
    masked_df = tt.mask_nans(tt.df)



    # print(tt.cross_vali_shufle(tt.df[tt.cechy], tt.df[tt.columns[0]],0.25, 0))
    #
    xtest, ytest, xtrain, ytrain = tt.test_data(masked_df[tt.cechy], masked_df[tt.columns[3]], train_size=0.1, r_state=10)
    est = Estimators(xtest, ytest, xtrain, ytrain)
    SVM_pred = est.SVM('rbf', 4)
    print(est.confusion_matrix(SVM_pred, 'AF', 'SVM'))
    bayes_pred = est.Bayes()
    print(est.confusion_matrix(bayes_pred, 'AF', "Bayes"))
    tree = est.DecisionTree()
    print(est.confusion_matrix(tree, 'AF', 'DecistionTree'))
    forest = est.Forest(n_est=1000)
    print(est.confusion_matrix(forest, 'AF', 'RandomForest'))