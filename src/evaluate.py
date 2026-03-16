import numpy as np
from sklearn.metrics import mean_squared_error


def evaluate(model, X_test, y_test):

    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)

    return mse, rmse