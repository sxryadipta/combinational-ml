from src.load_data import load_dataset
from src.models import bpnn, random_forest
from src.evaluate import evaluate


def run(train_path, test_path):

    X_train, y_train = load_dataset(train_path)
    X_test, y_test = load_dataset(test_path)

    bp = bpnn()
    rf = random_forest()

    bp.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    print("BPNN")
    print(evaluate(bp, X_test, y_test))

    print("Random Forest")
    print(evaluate(rf, X_test, y_test))