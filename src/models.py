from sklearn.ensemble import RandomForestRegressor

def get_rf():

    return RandomForestRegressor(
        n_estimators=1200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=True,
        random_state=42
    )