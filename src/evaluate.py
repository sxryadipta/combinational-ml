import numpy as np

def evaluate(model, X_test, y_test):

    pred = model.predict(X_test)

    # Correlation
    R = np.corrcoef(y_test, pred)[0, 1]

    # % Deviation
    deviation = np.mean(np.abs((y_test - pred) / y_test)) * 100

    return {
        "Correlation": R,
        "Deviation %": deviation
    }