import pandas as pd

def load_dataset(path):

    df = pd.read_csv(path)

    # ----- FEATURE ENGINEERING -----
    df["interaction"] = df["gates"] * df["inputs"]
    df["structure"] = df["VL"] * df["HL"]
    df["gate_density"] = df["gates"] / (df["VL"] * df["HL"] + 1e-6)
    df["power_proxy"] = df["gates"] * df["inputs"] * df["outputs"]

    features = [
        "VL", "HL", "gates", "inputs", "outputs",
        "interaction", "structure", "gate_density", "power_proxy"
    ]

    X = df[features].values
    y = df["power"].values

    return X, y