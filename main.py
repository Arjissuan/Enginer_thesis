import matplotlib.pyplot as plt
import pandas as pd
import os
from src.main_scripts.implemented import Implementations
from src.main_scripts.ML_implemented import ML_implemented


if __name__ == '__main__':
    analysis = Implementations(name_column='BomanIndex',
                               new_columns=['Cysteines', 'Asphartic Acid', 'Popular_aminoacids'],
                               count_amins=[['C', ], ['D'], ['G', 'L', 'K']],
                               rel_col='Length',
                               norm_cols=0)
    machinelern = ML_implemented(analysis.name_column)

    df = analysis.normalized_df[analysis.prep.get_cechy(analysis.normalized_df)]

    X_test, y_test, X_train, y_train = machinelern.train_test_split_feature_select(df=df, train_size=0.2, r_state=42, feature_select='None', normalizzation=False)
    predictions_matrix = machinelern.predictions(X_test, y_test, X_train, y_train)
    print(predictions_matrix)
    predictions_matrix.to_excel('./prediction_nonormandkbest.xlsx')

    X_test, y_test, X_train, y_train = machinelern.train_test_split_feature_select(df=df, train_size=0.2, r_state=42, feature_select='SelectKBest', n_feature=14, normalizzation=True)
    predictions_matrix = machinelern.predictions(X_test, y_test, X_train, y_train)
    print(predictions_matrix)
    predictions_matrix.to_excel('./prediction.xlsx')

    # making df with chosen columns for cross validation
    KBest_df = pd.concat([X_test, X_train], ignore_index=True)
    kbest_y = pd.concat([y_test, y_train], ignore_index=True)
    KBest_df['BomanIndex'] = kbest_y
    print(KBest_df[KBest_df['BomanIndex']==1])

    train_test_cross_matrix = machinelern.cross_validations(KBest_df, 10, 1)
    cross_vali_pred_matrix = machinelern.predictions_crossval(train_test_cross_matrix)
    print(cross_vali_pred_matrix)
    cross_vali_pred_matrix.to_excel('./crossvalidation.xlsx')