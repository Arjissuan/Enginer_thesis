from src.data_preparation.data_preparation import DataPreparation
from src.machine_learning.estimators import Estimators
from src.main_scripts.implemented import Implementations


if __name__ == '__main__':
    analysis = Implementations()
    normalized_df = analysis.dataframe_preparation(analysis.prep.get_dataset(),
                                   ['Cysteines', 'Small_aminoacids'],
                                   [['C',], ['G', 'L']])
    normalized_df.to_excel("./databases/normalized_dataset.xlsx")
# print(analysis.predictions(analysis.prep.get_dataset(), analysis.prep.columns[0], 'AB'))
#    analysis.clasterization_results()




