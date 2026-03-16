from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


def bpnn():

    model = MLPRegressor(
        hidden_layer_sizes=(15,15),
        activation='tanh',
        solver='lbfgs',
        max_iter=5000,
        random_state=42
    )

    return model


def random_forest():

    model = RandomForestRegressor(
        n_estimators=750,
        max_depth=15,
        min_samples_split=2,
        random_state=42
    )

    return model