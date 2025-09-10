import pandas as pd
import os


def main():
    ### Step 0: Initialize the file paths
    model = "InternVL3"
    temperature = "greedy"
    results_dir = f"../../results/phd/{model}/{temperature}"

    ### Step 0.1: Set the column names
    col_label = "label"
    col_response = "processed_response"
    col_task = "task"

    ### Step 1: Loop through the result files and judge
    for file in os.listdir(results_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(results_dir, file))
            # Set all col_response to lowercase and stripped
            df[col_response] = df[col_response].str.lower().str.strip()
            df[col_label] = df[col_label].str.lower().str.strip()
            # Create a "correct" column with values 0 or 1
            df["correct"] = (df[col_response].astype(str) == df[col_label].astype(str)).astype(int)
            # Get the stats of correctness rate per task
            stats = df.groupby(col_task)["correct"].mean().reset_index(name="correctness_rate")
            # Convert the correctness rate to percentage (times 100 and get one digit of decimal)
            stats["correctness_rate"] = (stats["correctness_rate"] * 100).round(1)
            print(f"File: {file}")
            print(stats[[col_task, "correctness_rate"]])

            # Write to the same csv file
            df.to_csv(os.path.join(results_dir, file), index=False)
            print(f"Updated file: {file}")
    


if __name__ == "__main__":
    main()