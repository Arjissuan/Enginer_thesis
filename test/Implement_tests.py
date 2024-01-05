from src.main_scripts.implemented import Implementations
from src.main_scripts.ML_implemented import ML_implemented

if __name__ == '__main__':
    analysis = Implementations('BomanIndex')
    machinelern = ML_implemented(analysis.name_column)

    # corrnorms = analysis.corelations_presentation(analysis.normalized_df, analysis.name_column, 2.48,True, True)
    # print(analysis.prep.corelations_rank(corrnorms[3], analysis.name_column))

    df = analysis.normalized_df[analysis.prep.get_cechy(analysis.normalized_df)]
    X_test, y_test, X_train, y_train = machinelern.train_test_split_feature_select(df=df, train_size=0.2, r_state=42,
                                                                                   feature_select='None',
                                                                                   normalizzation=False)
    svm_test = machinelern.test_various_SVM(X_test, y_test, X_train, y_train)
    print(svm_test)
    svm_test.to_excel('./test/SVM_NoNnormANDkbest.xlsx')

    mlp_test = machinelern.test_varoius_MLP(X_test, y_test, X_train, y_train)
    print(mlp_test)
    mlp_test.to_excel('./test/MLP_NONnormANDKbest_activ_10_5.xlsx')

    X_test, y_test, X_train, y_train = machinelern.train_test_split_feature_select(df=df, train_size=0.2, r_state=42,
                                                                                   feature_select='SelectKBest',
                                                                                   n_feature=14, normalizzation=True)
    svm_test = machinelern.test_various_SVM(X_test, y_test, X_train, y_train)
    print(svm_test)
    svm_test.to_excel('./test/SVM.xlsx')
    mlp_test = machinelern.test_varoius_MLP(X_test, y_test, X_train, y_train)
    print(mlp_test)
    mlp_test.to_excel('./test/MLP_activ_10_5.xlsx')