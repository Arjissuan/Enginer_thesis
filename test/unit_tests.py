from testy import Tests
from src.machine_learning.estimators import Estimators
import numpy as np

def clasterization_resutls():
    tt = Tests()
    for idx in range(len(tt.columns)):
        clastered = tt.clsterization(idx)
        auto_corr = lambda x: (x, tt.correlation(clastered[x]))
        correlated = dict(map(auto_corr, clastered.keys()))
        for key in correlated.keys():
            tt.heatmaps(correlated[key], key)

def test_masking_nans():
    tt = Tests()
    masked_df = tt.mask_nans(tt.chosen_AMP_df)
    masked = tt.mask_nans(tt.df)
    print(np.sum(masked_df.isnull().values.T, axis=1))
    print(np.sum(masked.isnull().values.T, axis=1))

if __name__ == "__main__":
    test_masking_nans()


    # print(tt.cross_vali_shufle(tt.df[tt.cechy], tt.df[tt.columns[0]],0.25, 0))
    #
