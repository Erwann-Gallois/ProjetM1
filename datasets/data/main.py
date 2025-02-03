from data.prepare_datasets import (
    prepare_daic_woz,
    prepare_damt_dataset,
    prepare_memo_dataset,
)

# ---- DAIC WOZ
daic_woz_path = "./data/DAIC-WOZ"
daic_woz_dataset = prepare_daic_woz(daic_woz_path)
print(daic_woz_dataset)

# ---- DAMT Dataset
damt_path = "./data/DAMT"
damt_dataset = prepare_damt_dataset(damt_path)
print(damt_dataset)

# ---- MEMO Dataset
memo_path = "./data/MEMO"
memo_dataset = prepare_memo_dataset(memo_path)
print(memo_dataset)

# Example usage:
print(daic_woz_dataset["dialogue"][0])
print(damt_dataset["train"]["dialogue"][0])
