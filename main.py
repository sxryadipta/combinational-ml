from src.run_experiment import run

print("\n----- NAND CIRCUITS -----")
nand_results = run("data/nand_train.csv", "data/nand_test.csv")
print(nand_results)

print("\n----- NOR CIRCUITS -----")
nor_results = run("data/nor_train.csv", "data/nor_test.csv")
print(nor_results)