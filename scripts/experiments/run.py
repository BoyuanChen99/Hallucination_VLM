import pandas as pd
import os
import sys
import argparse
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.abspath(os.path.join(root_dir, 'src')))
from vlms import init_vlm
from utils import load_dataframe, concatenate_response, disable_huggingface_warnings


args = argparse.ArgumentParser(description="Run InternVL3 for HaloQuest evaluation")
# Experiment Group and Category
args.add_argument("--model", type=str, default="InternVL3", help="Model name")
args.add_argument("--dataset", type=str, default="chair", help="The dataset to run test on")
# Hyperparameters
args.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
args.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
args.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling")
FLAGS = args.parse_args()


def main(args):
    ### Step 0: Initialize the vlm, the dataset file, and the output file
    disable_huggingface_warnings()
    vlm = init_vlm(args.model)
    dataset = args.dataset
    dataset_dir = "../../../data"
    df, col_prompt, col_image, image_dir = load_dataframe(dataset, dataset_dir)
    output_dir = f"../../results"
    output_file = os.path.join(output_dir, f"{dataset}_{args.model}.csv")
    if os.path.exists(output_file):
        df_output = pd.read_csv(output_file)
    else:
        df_output = pd.DataFrame(columns=["idx_image", "idx_question"] + df.columns.tolist() + ["response"])

    ### Step 1: Loop through the dataset and do inference
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        ### Step 1.1: Continue if df_output already has a row matching the row's col_prompt and col_image
        if (
            not df_output.empty
            and col_prompt in df_output.columns
            and col_image in df_output.columns
        ):
            mask = (df_output[col_prompt] == row[col_prompt]) & (df_output[col_image] == row[col_image])
            if mask.any():
                continue

        ### Step 1.2: Prepare the prompt and the image path
        prompt = row[col_prompt]
        image_path = os.path.join(image_dir, row[col_image])
        
        ### Step 1.3: Infer
        response = vlm.infer(
                        prompt=prompt, 
                        image_path=image_path,
                        max_new_tokens=512,
                        temperature=0,
                    )

        ### Step 1.4: Concatenate to the end of df_output
        df_output = concatenate_response(response, row, df_output, col_image)

        ### Step 1.5: Write to the output path
        with open(output_file, "w") as f:
            df_output.to_csv(f, index=False)

    

if __name__ == "__main__":
    main(FLAGS)