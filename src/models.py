from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


def bpnn():

    model = MLPRegressor(
        hidden_layer_sizes=(15,15),
        activation='tanh',
        solver='sgd',
        learning_rate_init=0.25,
        max_iter=750,
        random_state=42
    )

    return model


def random_forest():

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        random_state=42
    )

    return model