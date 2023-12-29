import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preparation.data_preparation import DataPreparation

class Implementations:
    def __init__(self, name_column):
        self.name_column = name_column
        self.prep = DataPreparation(dataset_dest='./databases/AMP_30_03_2020_IMPROVED.xlsx',
                                    clst_columns=["A_BACT", "A_VIRAL", 'A_CANCER', 'A_FUNGAL', 'RANGE'],
                                    figure_dest='./figures',
                                    columns_to_drop=['Age Tre of life', 'Radius_gyration', 'Abrev.'])
        self.normalized_df = self.dataframe_preparation(self.prep.get_dataset(),
                                                        ['Cysteines', 'Small_aminoacids'],
                                                        [['C',], ['G', 'L']],
                                                        rel_column='Length')

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

    def repetition_results(self, sequences: pd.Series, lengths: pd.Series) -> pd.DataFrame:
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
        repeted['Repetitions_length'] = list(repetition_length)
        repeted['Peptide_length'] = lengths.values
        repeted['Repetitions_percentages'] = self.prep.data_normalization(repeted, 'Peptide_length', ['Repetitions_length',])
        repeted['Number_of_repe'] = list(map(lambda x: len(x), repeted['Listed_Repetitions']))
        return repeted


    def clasterization_results(self, df, size=(40,35), font_scale=1.5, fmt='.1f', annota=7, save=False):
        cechy = self.prep.get_cechy(df)
        for indx in range(len(self.prep.clst_columns)):
            claster = self.prep.clsterization(indx, df)
            auto_corr = lambda x: (x, self.prep.correlation(claster[x]))
            corrls = dict(map(auto_corr, claster.keys()))
            for k in corrls.keys():
                self.prep.heatmaps(corrls[k],
                                   cechy=cechy,
                                   title=k,
                                   size=size,
                                   font_scale=font_scale,
                                   fmt=fmt,
                                   annota=annota,
                                   save=save)


    def df_corelations_drop(self, df: pd.DataFrame, column: str, corr_level: float) -> pd.DataFrame:
        cechy = self.prep.get_cechy(df)
        corrmat = self.prep.correlation(df=df)
        indx = list(i for i in range(len(cechy)) if cechy[i] == column)[0]
        corelated = list(cechy[i] for i,c in enumerate(corrmat[indx]) if c > corr_level)
        return df.drop(columns=corelated, axis=1)

    def corelations_presentation(self, df: pd.DataFrame, column: str, level: float = 2.48, heatmaps=False, save_heatm=False, save_df=False):
        corrmat = self.prep.correlation(df)
        matrix_low = self.prep.correlation(df[df[column] <= level])
        matrix_high = self.prep.correlation(df[df[column] > level])
        cechy = self.prep.get_cechy(df)
        crmm =  pd.DataFrame(data=corrmat, columns=cechy, index=cechy)
        crmm_low= pd.DataFrame(data=matrix_low, columns=cechy, index=cechy)
        crmm_high = pd.DataFrame(data=matrix_high, columns=cechy, index=cechy)
        roznica = ((crmm_low - crmm_high)**2)**(1/2)
        if heatmaps is True:
            titles = ['Heatmap of corelations between columns',
                      f'Heatmap of corelations when {column} values are smaller than {level}',
                      f'Heatmap of corelations when {column} values are bigger than {level}',
                      f'Heatmap of change in corelations between {column} level value of {level}']
            for indx, matrix in enumerate([crmm, crmm_low, crmm_high, roznica]):
                self.prep.heatmaps(matrix, matrix.columns, titles[indx], save=save_heatm)
        if save_df is True:
            for matrix in (crmm, crmm_low, crmm_high, roznica):
                matrix.to_excel(input('Corelation matrix destination:'))
        return crmm, crmm_low, crmm_high, roznica

    def corelations_histograms(self, corrmatrix, save=False):
        for col in corrmatrix.columns:
            sns.histplot(data=corrmatrix[col], bins=20)
            plt.xlim(0, 1)
            plt.ylim(0, 10)
            plt.show()
            if save is True:
                plt.savefig(f'./figures/corelation_histograms/{col}')
