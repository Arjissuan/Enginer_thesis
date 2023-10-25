import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ShuffleSplit, train_test_split

class Tests:
    def __init__(self):
        self.df = pd.read_excel('../databases/AMP_30_03_2020_IMPROVED.xlsx').drop(columns=['Kolumna1'])
        self.columns = ["A_BACT", "A_VIRAL", 'A_CANCER', 'A_FUNGAL', 'RANGE'] #'Abrev.', has to be done
        self.cechy = self.df.describe().columns

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
        for col in hot_cols:
            column = df[col]
            new_df = df.drop(columns=[col])
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

    def cross_vali_shufle(self, x_df, y_df, train_s, r_state):
        shuffle = ShuffleSplit(n_splits=10, train_size=train_s, random_state=r_state)
        for train_indx, test_indx in shuffle.split(X=x_df, y=y_df):
            print(train_indx, test_indx)

    def test_data(self, x, y, train_size, r_state):
        X_train, X_test, y_train, y_test = train_test_split(x,y ,train_size=train_size, random_state=r_state)
        return X_train, y_train, X_test, y_test