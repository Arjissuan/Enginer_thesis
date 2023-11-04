import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections.abc import Generator
from sklearn.model_selection import ShuffleSplit, train_test_split

class Tests:
    def __init__(self):
        self.df = pd.read_excel('../databases/AMP_30_03_2020_IMPROVED.xlsx').drop(columns=['Kolumna1', 'Age Tre of life', 'Radius_gyration', 'Abrev.'])
        self.columns = ["A_BACT", "A_VIRAL", 'A_CANCER', 'A_FUNGAL', 'RANGE'] #'Abrev.', has to be done
        self.cechy = self.df.describe().columns
        self.chosen_AMP_df = pd.read_excel('../databases/Final_Selected_Diverse_AMPs.xlsx').drop(columns=['Kolumna1', 'Age Tre of life', 'Radius_gyration', 'Abrev.'])

    def clsterization(self, index):
        item = self.columns[index]
        values_set = tuple(set(self.df[item]))
        df_list = dict(map(lambda x: (f'{item}_{x}', self.df[self.df[item] == x]), values_set))
        return df_list

    def correlation(self, df):
        datarrays = df[self.cechy].values.T
        m_df = np.ma.masked_invalid(datarrays)
        corr = np.ma.corrcoef(m_df)
        return corr

    def heatmaps(self, df, title, size=(40,35), font_scale=1.5, fmt='.1f', annota=7):
        plt.figure(figsize=size)
        plt.title(title)
        sns.set(font_scale=font_scale, )
        sns.heatmap(df, cbar=True, annot=True,
                    square=True, fmt=fmt,
                    annot_kws={'size': annota},
                    xticklabels=self.cechy,
                    yticklabels=self.cechy)
        plt.show()

    def one_hot_encoder(self, hot_cols: tuple[str], df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.drop(columns=hot_cols[0])

        for col in hot_cols:
            column = df[col]
            column_v_set = list(set(column))
            true_set = []

            for item in column_v_set:
                if '/' in item:
                    new_item = item.replace('"', '').split('/')
                else:
                    new_item = item.replace('"', '').split(',')
                true_set = true_set + new_item

            new_col_vset = list(set(true_set))
            is_value_in = lambda x: (1 if value in column[x] else 0)
            new_columns = pd.DataFrame()
            for value in new_col_vset:
                bin_col = list(map(is_value_in, range(len(column))))
                new_columns[value] = bin_col

            new_df = pd.concat([new_df, new_columns], axis=1)

        return new_df

# this function is still in development
    def cross_vali_shufle(self, x_df, y_df, train_s, r_state):
        shuffle = ShuffleSplit(n_splits=10, train_size=train_s, random_state=r_state)
        for train_indx, test_indx in shuffle.split(X=x_df, y=y_df):
            # print(train_indx, test_indx)
            # print(x_df.iloc[train_indx, :])
            print(y_df.iloc[train_indx])

    def test_data(self, x, y, train_size, r_state):
        X_train, X_test, y_train, y_test = train_test_split(x,y ,train_size=train_size, random_state=r_state)
        return X_test, y_test, X_train, y_train

    def flatten(self, lista):
        if isinstance(lista, (list, tuple)):
            for item in lista:
                for l in self.flatten(item):
                    yield l
        else:
            yield lista

    def mask_nans(self, df):
        new_df = df.isnull()
        indexing_nans = lambda x: (list(new_df[new_df[x] == True].index) if 1 in new_df[x] else np.nan)
        indexes = list(map(indexing_nans, new_df.columns))
        nans_indexes = []
        all_indexes = lambda x: (nans_indexes.append(indexes[x]) if len(indexes[x]) > 0 and indexes[x] not in nans_indexes else 0)
        list(map(all_indexes, range(len(indexes))))
        nans_indx = list(set(self.flatten(nans_indexes)))
        return df.drop(index=nans_indx) #have to change indexing after this

    def data_normalization(self, df, rel_col, cols_to_perc):

        def generate_cols(data, relcol, num_of_percol):
            for i in num_of_percol:
                genereted_cols = data.loc[:, relcol]
                yield genereted_cols

        cols_list = list(generate_cols(df, rel_col, range(len(cols_to_perc))))
        context_cols = pd.DataFrame(cols_list).values
        chosen_cols = df.loc[:, cols_to_perc].values.T
        normalied_cols = np.divide(np.multiply(chosen_cols, 100), context_cols).T
        return pd.DataFrame(data=normalied_cols, columns=cols_to_perc)

    def count_amino_acids(self, df: pd.Series, amiacids: list[str]) -> Generator[int]:
        if len(amiacids) == 1:
            count_amiacid = lambda x: (1 if amiacids[0] == x else 0)
            for seqence in df:
                yield np.sum(list(map(count_amiacid, seqence)))
        elif len(amiacids) > 1:
            count_amiacid = lambda x: (1 if x in amiacids else 0)
            for seqence in df:
                yield np.sum(list(map(count_amiacid, seqence)))
