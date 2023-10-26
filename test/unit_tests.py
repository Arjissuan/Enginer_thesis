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
    masked_df = (tt.mask_nans(tt.df))


    # print(tt.cross_vali_shufle(tt.df[tt.cechy], tt.df[tt.columns[0]],0.25, 0))
    #
    xtest, ytest, xtrain, ytrain = tt.test_data(masked_df[tt.cechy], masked_df[tt.columns[0]], train_size=0.3, r_state=10)
    est = Estimators(xtest, ytest, xtrain, ytrain)
    SVM_pred = est.SVM('rbf', 4)
    print(est.efficiency(SVM_pred))
    bayes_pred = est.Bayes()
    print(est.efficiency(bayes_pred))
    tree = est.DecisionTree()
    print(est.efficiency(tree))
    forest = est.Forest(n_est=100)
    print(est.efficiency(forest))