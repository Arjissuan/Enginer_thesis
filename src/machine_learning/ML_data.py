from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, StratifiedKFold, GridSearchCV
import numpy as np
import pandas as pd

class ML_data:
    def __init__(self, class_vector):
        self.Class_vector = class_vector

    def creating_bina_class(self, df, column, value=2.48):
        vector = df[column].values
        return np.logical_not(vector < value).astype(np.int8)

    def test_data(self, x: pd.DataFrame,
                  y: pd.Series,
                  train_size: float,
                  r_state: float) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=r_state)
        return X_test, y_test, X_train, y_train

    def cross_vali_shufle(self, x_df: np.array, y_df: np.array, test_s: float, r_state: float) -> np.array:
        shuffle = StratifiedShuffleSplit(n_splits=10, test_size=test_s, random_state=r_state)
        shuf_df = shuffle.split(X=x_df, y=y_df)
        return shuf_df

    def one_hot_encoder(self, hot_cols: tuple[str], df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.drop(columns=[hot_cols])

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

    def Pareto_Scaling(self, vector_values: np.ndarray) -> np.ndarray:
        stnd_dev = np.std(vector_values)
        mean = np.mean(vector_values)
        PS = list(map(lambda x: (x - mean)/stnd_dev**(1/2), vector_values))
        return np.array(PS)