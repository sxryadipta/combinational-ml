from src.run_experiment import run


print("\n----- NAND CIRCUITS -----")

run(
    "data/nand_train.csv",
    "data/nand_test.csv"
)


print("\n----- NOR CIRCUITS -----")

run(
    "data/nor_train.csv",
    "data/nor_test.csv"
)