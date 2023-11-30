import pandas as pd
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


def data_normalization_test():
    tt = Tests()
    tt.data_normalization(tt.chosen_AMP_df,
                          'Length',
                          [
                              'Aliphatic',
                              'Aromatic',
                              'NonPolar',
                              'Polar',
                              'Charged',
                              'Basic',
                              'Acidic',
                          ])

    tt.data_normalization(tt.df,
                          'Length',
                          [
                              'Aliphatic',
                              'Aromatic',
                              'NonPolar',
                              'Polar',
                              'Charged',
                              'Basic',
                              'Acidic',
                          ]
                          )

def test_count_aminoacis():
    tt = Tests()
    df = tt.mask_nans(tt.chosen_AMP_df)
    df2 = tt.mask_nans(tt.df)
    print(list(tt.count_amino_acids(df['Sequence'], ['G', 'L'])))
    print(list(tt.count_amino_acids(df['Sequence'], ['C'])))
    print(list(tt.count_amino_acids(df2['Sequence'], ['G', 'L'])))
    print(list(tt.count_amino_acids(df2['Sequence'], ['C'])))

def test_data_prep(): #change indexing of masking nans
    new_cols = ['Cysteines', 'Small_aminoacids']
    columns = ['Aliphatic',
               'Aromatic',
               'NonPolar',
               'Polar',
               'Charged',
               'Basic',
                'Acidic',]
    new_df = tt.mask_nans(tt.df)
    for indx,item in enumerate([['C',], ['G', 'L']]):
        vector = np.asarray(list(tt.count_amino_acids(new_df['Sequence'], item)), dtype=np.int32)
        new_df[new_cols[indx]] = vector
    normalized_df = tt.data_normalization(new_df, 'Length', new_cols)
    normalized_df2 = tt.data_normalization(new_df, 'Length', columns)
    data_frame = new_df.drop(columns=columns+new_cols)
    return pd.concat([normalized_df, normalized_df2, data_frame], axis=1)

def repetiton_test():
    tt = Tests()
    print(tt.repetitions('QCCCCGSLGGGLLCQV'))
    print(tt.repetitions('CCGHQC'))
    print(tt.repetitions('CLGGGLLLQVKYYY'))
    print(tt.repetitions('UUUUUGGCCCLCCCCJKUUUUOPQUUUUU'))
    print(tt.repetitions('DDDDDDD'))

def corelations_drop(column, corr_level, corrmat_or_df):
    tt = Tests()
    df = test_data_prep()
    cechy = df.describe().columns
    corrmat = tt.correlation(df=df)
    indx = tuple(i for i, item in enumerate(cechy) if item == column)[0]
    corr_vecor = corrmat[:, indx]
    unrelated_cols = [i for i, item in enumerate(corr_vecor) if item < corr_level]
    firstmatr = corrmat[unrelated_cols, :]
    secodn_matr = firstmatr[:, unrelated_cols]
    if corrmat_or_df == 'df':
        return df.iloc[:, unrelated_cols]
    else:
        return secodn_matr


if __name__ == "__main__":
    # test_masking_nans()
    # data_normalization_test()
    tt = Tests()
    # print(tt.cross_vali_shufle(tt.df[tt.cechy], tt.df[tt.columns[0]],0.25, 0))
    #repetiton_test()
    #corelations_drop('BomanIndex', 0.5, 'df')
