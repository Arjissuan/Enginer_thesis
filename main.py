from src.data_preparation.data_preparation import DataPreparation
from src.machine_learning.estimators import Estimators
from src.main_scripts.implemented import Implementations


if __name__ == '__main__':
    analysis = Implementations()
    print(analysis.predictions(analysis.prep.get_dataset(), analysis.prep.columns[0], 'NOT'))
#    analysis.clasterization_results()



