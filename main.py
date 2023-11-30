import pandas as pd

from src.main_scripts.implemented import Implementations
from src.main_scripts.ML_implemented import ML_implemented


if __name__ == '__main__':
    analysis = Implementations('BomanIndex')
    machinelern = ML_implemented(analysis.name_column)

    corrnorms = analysis.corelations_presentation(analysis.normalized_df, analysis.name_column, 2.48)
    print(analysis.corelations_rank(corrnorms[3], analysis.name_column))
    # analysis.clasterization_results()
    # print(analysis.predictions(analysis.prep.get_dataset(), analysis.prep.columns[0], 'AB'))
    df_machine_learning = analysis.normalized_df[analysis.prep.get_cechy(analysis.normalized_df)]
    print(machinelern.predictions(df=df_machine_learning))