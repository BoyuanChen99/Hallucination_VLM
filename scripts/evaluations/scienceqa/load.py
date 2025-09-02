from datasets import load_dataset

# Load the dataset
dataset = load_dataset("lmms-lab/ScienceQA-IMG")

# Print the number of examples in the test split
print("Number of test examples:", len(dataset["test"]))
