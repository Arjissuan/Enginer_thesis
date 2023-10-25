from testy import Tests

def clasterization_resutls():
    tt = Tests()
    for idx in range(len(tt.columns)):
        clastered = tt.clsterization(idx)
        auto_corr = lambda x: (x, tt.correlation(clastered[x]))
        correlated = dict(map(auto_corr, clastered.keys()))
        for key in correlated.keys():
            tt.heatmaps(correlated[key], key)


if __name__ == "__main__":
    tt = Tests()
    print(tt.cross_vali_shufle(tt.df[tt.cechy[0]], tt.df[tt.columns], 0.25, 0))


