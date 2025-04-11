
from datasets import concatenate_datasets, load_dataset
import os

def main():
    # load dataset
    dataset_1 = load_dataset("json" , data_files= "finance_data.json")  


    # create a column called text
    dataset_1 = dataset_1.map(
        lambda example: {"text": example["instruction"] + " " + example["output"]},
        num_proc=4,
    )
    dataset_1 = dataset_1.remove_columns(["input", "instruction", "output"])


    with open("financial_data.txt", "w", encoding="utf-8") as file:
        for example in dataset_1["train"]:
            file.write(example['text'] + "\n")

if __name__ == "__main__":
    main()
