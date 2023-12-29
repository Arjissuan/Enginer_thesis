from src.main_scripts.implemented import Implementations
from src.main_scripts.ML_implemented import ML_implemented


if __name__ == '__main__':
    analysis = Implementations('BomanIndex')
    machinelern = ML_implemented(analysis.name_column)

    corrnorms = analysis.corelations_presentation(analysis.normalized_df, analysis.name_column, 2.48,True, True)
    print(analysis.prep.corelations_rank(corrnorms[3], analysis.name_column))


    df = analysis.normalized_df[analysis.prep.get_cechy(analysis.normalized_df)]

    X_test, y_test, X_train, y_train = machinelern.train_test_split_feature_select(df=df, train_size=0.26, r_state=1)
    predictions_matrix = machinelern.predictions(X_test, y_test, X_train, y_train)
    print(predictions_matrix)
    predictions_matrix.to_excel('./prediction.xlsx')

    train_test_cross_matrix = machinelern.cross_validations(df, 10, 42)
    cross_vali_pred_matrix = machinelern.predictions_crossval(train_test_cross_matrix)
    print(cross_vali_pred_matrix)
    cross_vali_pred_matrix.to_excel('./crossvalidation.xlsx')