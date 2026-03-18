from src.load_data import load_dataset        
from src.models import get_rf
from src.evaluate import evaluate

from sklearn.model_selection import GridSearchCV


def run(train_path, test_path):

    X_train, y_train = load_dataset(train_path)
    X_test, y_test = load_dataset(test_path)

    param_grid = {
        "n_estimators": [500, 800, 1200],
        "max_depth": [10, 15, None],
        "min_samples_split": [2, 3]
    }

    grid = GridSearchCV(
        get_rf(),
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error"
    )

    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    print("Best Params:", grid.best_params_)

    results = evaluate(model, X_test, y_test)

    return results