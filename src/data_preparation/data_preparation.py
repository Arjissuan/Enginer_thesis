import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterator
import seaborn as sns
import os


class DataPreparation:
    def __init__(self, dataset_dest, clst_columns, figure_dest, columns_to_drop):
        self.dataset_dest = dataset_dest
        self.clst_columns = clst_columns  # 'Abrev.', has to be done
        self.figure_dest = figure_dest
        self.columns_to_drop = columns_to_drop
        # self.chosen_data = './databases/Final_Selected_Diverse_AMPs.xlsx'

    def get_dataset(self):
        df = pd.read_excel(self.dataset_dest, index_col=0).drop(columns=self.columns_to_drop)
        return df

    def get_cechy(self, df):
        return df.describe().columns

    def clsterization(self, index, df): #change it to more ogolna
        item = self.clst_columns[index]
        values_set = tuple(set(df[item]))
        df_list = dict(map(lambda x: (f'{item}_{x}', df[df[item] == x]), values_set))
        return df_list

    def correlation(self, df):
        datarrays = df[self.get_cechy(df)].values.T
        m_df = np.ma.masked_invalid(datarrays)
        corr = np.ma.corrcoef(m_df)
        return corr

    def heatmaps(self, corld_df, cechy, title, size=(40, 35), font_scale=3, fmt='.1f', annota=24, save=False):
        plt.figure(figsize=size)
        plt.title(title)
        sns.set(font_scale=font_scale, )
        sns.heatmap(corld_df, cbar=True, annot=True,
                    square=True, fmt=fmt,
                    annot_kws={'size': annota},
                    xticklabels=cechy,
                    yticklabels=cechy)
        if save is True:
            name = f'{title}.pdf'
            plt.savefig(os.path.join(self.figure_dest, name))
        plt.show()


    def mask_nans(self, df: pd.DataFrame, threshold=0.05): ###zmienic aby usuwalo kolumny powyzej pewnej wartosci brakujacych rekordow
        new_df = df.isnull()
        indexing_nans = lambda x: (list(new_df[new_df[x] == True].index) if 1 in new_df[x] else np.nan)
        indexes = list(map(indexing_nans, new_df.columns))
        nans_indexes = []
        all_indexes = lambda x: (
            nans_indexes.append(indexes[x]) if len(indexes[x]) > 0 and indexes[x] not in nans_indexes else 0)
        list(map(all_indexes, range(len(indexes))))
        nans_indx = list(set(self.flatten(nans_indexes)))
        return df.drop(index=nans_indx).reset_index(drop=True)

    def flatten(self, lista: list) -> list:
        if isinstance(lista, (list, tuple)):
            for item in lista:
                for l in self.flatten(item):
                    yield l
        else:
            yield lista

    def data_normalization(self, df:pd.DataFrame, rel_col: str, cols_to_perc: list[str]) -> pd.DataFrame:
        # returns specified normalized columns
        def generate_cols(data, relcol, num_of_percol):
            for i in num_of_percol:
                genereted_cols = data.loc[:, relcol]
                yield genereted_cols

        cols_list = list(generate_cols(df, rel_col, range(len(cols_to_perc))))
        context_cols = pd.DataFrame(cols_list).values
        chosen_cols = df.loc[:, cols_to_perc].values.T
        normalied_cols = np.divide(np.multiply(chosen_cols, 100), context_cols).T

        return pd.DataFrame(data=normalied_cols, columns=cols_to_perc)

    def count_amino_acids(self, seria: pd.Series, amiacids: list[str]) -> Iterator[int]:
        # function use one list of aminoacids that we want to count in sequence across all dataframe
        # it returns column with counts of specified aminoacid
        if len(amiacids) == 1:
            count_amiacid = lambda x: (1 if amiacids[0] == x else 0)
            for seqence in seria:
                yield np.sum(list(map(count_amiacid, seqence)))
        elif len(amiacids) > 1:
            count_amiacid = lambda x: (1 if x in amiacids else 0)
            for seqence in seria:
                yield np.sum(list(map(count_amiacid, seqence)))

    def repetitions(self, sequence):
        # return vectors of repetitions
        repeted_amins = {}
        # making vector for every index of next positions with marking if next positions have the same aminoacid or not
        for i, aminoacid in enumerate(sequence):
            repeted_amins[i] = []
            for j in range(i, len(sequence)):
                if aminoacid == sequence[j]:
                    repeted_amins[i].append(1)
                else:
                    repeted_amins[i].append(0)
        # print(repeted_amins)
        # making map of every center of repetition
        repe_centers = [0]*len(sequence)
        for i, vectors in enumerate(repeted_amins.values()):
            if len(vectors) >= 3:
                for j in range(1, len(list(vectors)) - 1):
                    if vectors[j] == 1 and vectors[j - 1] == 1 and vectors[j + 1] == 1:
                        repe_centers[i+j] = sequence[i+j]
        # print(repe_centers)
        # changing map of repetition centers into listed repetitions
        listed_repets = ''
        for indx, amino in enumerate(repe_centers):
            if indx < len(repe_centers)-1:
                if amino == 0 and repe_centers[indx + 1] != 0:
                    listed_repets += repe_centers[indx + 1]
            if amino != 0:
                listed_repets+=amino
            if amino == 0 and repe_centers[indx-1]!= 0 :
                listed_repets+=f'{repe_centers[indx-1]}_'

        listed_repets = listed_repets.split('_')
        return listed_repets[0:len(listed_repets)-1]

    def corelations_rank(self, matrix, column): #zwraca ranking najwiekszych zmian korelacji/najwiekszych korelacji.
        vector = matrix[column].copy()
        array = {}
        while len(vector) != 0:
            cell = vector[vector == np.max(vector)]
            array[str(cell.index[0])] = cell.values[0] #futurewarning
            vector = vector.drop(index=cell.index, axis=1)
        return pd.Series(array)