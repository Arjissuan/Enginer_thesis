from src.data_preparation.data_preparation import DataPreparation
from src.machine_learning.estimators import Estimators

class Implementations:
    def __init__(self):
        self.prep = DataPreparation()

    def clasterization_results(self, df, size=(40,35), font_scale=1.5, fmt='.1f', annota=7):
        for indx in range(len(self.prep.columns)):
            claster = self.prep.clsterization(indx, df)
            auto_corr = lambda x: (x, self.prep.correlation(claster[x]))
            corrls = dict(map(auto_corr, claster.keys()))
            for k in corrls.keys():
                self.prep.heatmaps(corrls[k], title=k, size=size, font_scale=font_scale, fmt=fmt, annota=annota)