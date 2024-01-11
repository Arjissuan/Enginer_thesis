import pandas as pd
import os
from src.data_preparation.data_preparation import DataPreparation

class Implementations:
    def __init__(self, name_column, new_columns, count_amins, rel_col, norm_cols=0):
        self.name_column = name_column
        self.prep = DataPreparation(dataset_dest='./databases/AMP_30_03_2020_IMPROVED.xlsx',
                                    columns_to_drop=['Age Tre of life', 'Radius_gyration', 'Abrev.'])
        self.normalized_df = self.dataframe_preparation(self.prep.get_dataset(),
                                                        new_columns,
                                                        count_amins,
                                                        rel_col,
                                                        norm_cols=norm_cols)
    def dataframe_preparation(self, df: pd.DataFrame,
                              new_columns: list[str],
                              count_amins: list[list[str]],
                              rel_column: str,
                              norm_cols=0
                              ) -> pd.DataFrame:

        if norm_cols == 0:
            standard_columns = ['Aliphatic',
                                'Aromatic',
                                'NonPolar',
                                'Polar',
                                'Charged',
                                'Basic',
                                'Acidic', ]
        elif norm_cols == 1:
            standard_columns = list(input('Specify columns to normalization: '))
        else:
            standard_columns = None

        masked_df = self.prep.mask_nans(df)
        vectors = pd.DataFrame(dict(map(lambda x, y: (x, list(self.prep.count_amino_acids(masked_df['Sequence'], y))),
                                        new_columns, count_amins)))
        new_df = pd.concat([masked_df, vectors], axis=1)

        normalized_df = self.prep.data_normalization(new_df, rel_column, new_columns)
        if standard_columns != None:
            normalized_df2 = self.prep.data_normalization(new_df, rel_column, standard_columns)
            data_frame = new_df.drop(columns=standard_columns + new_columns)
            return pd.concat([normalized_df, normalized_df2, data_frame], axis=1)
        else:
            data_frame = new_df.drop(columns=new_columns)
            return pd.concat([normalized_df, data_frame], axis=1)


    def repetition_results(self, sequences: pd.Series) -> pd.DataFrame:
        def generate_repetions(seqs):
            for s in seqs:
                vector = self.prep.repetitions(s)
                yield (vector, ''.join(vector))

        code_letters = ['A', 'G', 'I', 'L', 'P', 'V', 'F', 'W', 'Y', 'D', 'E', 'R', 'H', 'K', 'S', 'T', 'C', 'M',
                        'N' 'Q']
        repeted = pd.DataFrame(data=list(generate_repetions(sequences)),
                               columns=['Listed_Repetitions', 'Total_repetitions']
                               )
        repetition_length = self.prep.count_amino_acids(seria=repeted['Total_repetitions'], amiacids=code_letters)
        lengths = list(map(lambda x: len(x), sequences))
        repeted['Total_Sequence'] = sequences
        repeted['Repetitions_length'] = list(repetition_length)
        repeted['Peptide_length'] = lengths
        repeted['Repetitions_percentages'] = self.prep.data_normalization(repeted, 'Peptide_length', ['Repetitions_length',])
        repeted['Number_of_repe'] = list(map(lambda x: len(x), repeted['Listed_Repetitions']))
        return repeted


    def corelations_presentation(self, df: pd.DataFrame, column: str, level: float = 2.48, heatmaps=False, save_heatm=False, save_df=False, titles='non_custom'):
        if titles == 'non_custom':
            titles = ['Heatmap of corelations between columns',
                      f'Heatmap of corelations when {column} values are smaller than {level}',
                      f'Heatmap of corelations when {column} values are bigger than {level}',
                      f'Heatmap of change in corelations between {column} level value of {level}']

        corrmat = self.prep.correlation(df)
        matrix_low = self.prep.correlation(df[df[column] <= level])
        matrix_high = self.prep.correlation(df[df[column] > level])
        cechy = self.prep.get_cechy(df)
        crmm =  pd.DataFrame(data=corrmat, columns=cechy, index=cechy)
        crmm_low= pd.DataFrame(data=matrix_low, columns=cechy, index=cechy)
        crmm_high = pd.DataFrame(data=matrix_high, columns=cechy, index=cechy)
        roznica = ((crmm_low - crmm_high)**2)**(1/2)
        if heatmaps is True:
            for indx, matrix in enumerate([crmm, crmm_low, crmm_high, roznica]):
                self.prep.heatmaps(matrix,
                                   cechy=matrix.columns,
                                   title=titles[indx],
                                   size=(30,25),
                                   font_scale=1.9,
                                   fmt='.2f',
                                   annota=10,
                                   save=save_heatm)
        if save_df is True:
            for i, matrix in enumerate([crmm, crmm_low, crmm_high, roznica]):
                matrix.to_excel(os.path.join(os.getcwd(), './databases/wyniki/', f'{titles[i]}.xlsx'))

        return crmm, crmm_low, crmm_high, roznica
