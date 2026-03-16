import numpy as np
from sklearn.metrics import mean_squared_error


def evaluate(model, X_test, y_test):

    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)

    # correlation
    R = np.corrcoef(y_test, pred)[0,1]

    # percentage deviation
    deviation = np.mean(np.abs((y_test - pred) / y_test)) * 100

    return {
        "RMSE": rmse,
        "Correlation": R,
        "Deviation %": deviation
    }