import pandas as pd

def load_dataset(path):

    df = pd.read_csv(path)

    # ----- FEATURE ENGINEERING -----
    df["interaction"] = df["gates"] * df["inputs"]
    df["structure"] = df["VL"] * df["HL"]
    df["gate_density"] = df["gates"] / (df["VL"] * df["HL"] + 1e-6)

    features = [
        "VL", "HL", "gates", "inputs", "outputs",
        "interaction", "structure", "gate_density"
    ]

    X = df[features].values
    y = df["power"].values

    return X, y