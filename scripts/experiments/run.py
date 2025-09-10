import pandas as pd
import os
import sys
import argparse
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.abspath(os.path.join(root_dir, 'src')))
from vlms import init_vlm
from utils import load_dataframe, process_response, concatenate_response, disable_huggingface_warnings


args = argparse.ArgumentParser(description="Run InternVL3 for HaloQuest evaluation")
# VLM Model
args.add_argument("--model", type=str, default="InternVL3", help="Model name")
# Dataset and subgroup
args.add_argument("--dataset", type=str, default="phd", help="The dataset to run test on")
args.add_argument("--prompt_file", type=str, default="phd.txt", help="Path to the prompt file other than the benchmark's prompt")
args.add_argument("--subset", type=str, default="base", help="pope: {gqa, aokvqa, coco}; phd: {base, icc, iac(sec), ccs}")
args.add_argument("--subsplit", type=str, default="popular", help="Only used for POPE dataset. Choices are: {popular, adversarial, random}")
# Hyperparameters
args.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. When set to 0.0 the model will do greedy decoding and ignore the other parameters. ")
args.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
args.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling")
FLAGS = args.parse_args()


def main(args):
    ### Step 0: Initialize the vlm, the dataset file, and the output file
    # Initialize VLM
    disable_huggingface_warnings()
    print(f"Initializing {args.model}...")
    vlm = init_vlm(args.model)
    print(f"{args.model} VLM loaded.")
    # Input file and df (calling load_dataframe in utils)
    dataset = args.dataset
    dataset_dir = "../../../data"
    df, col_prompt, col_image, image_dir = load_dataframe(dataset, dataset_dir, subset=args.subset, subsplit=args.subsplit)
    # Output file
    output_dir = f"../../results"
    output_data_dir = os.path.join(output_dir, dataset)
    os.makedirs(output_data_dir, exist_ok=True)
    output_model_dir = os.path.join(output_data_dir, args.model)
    os.makedirs(output_model_dir, exist_ok=True)
    temp_str = f"temp{args.temperature}" if args.temperature > 0.0 else "greedy"
    output_temp_dir = os.path.join(output_model_dir, temp_str)
    os.makedirs(output_temp_dir, exist_ok=True)
    if "pope" in args.dataset:
        output_file = os.path.join(output_temp_dir, f"{args.subset}_{args.model}_{args.subsplit}.csv")
    elif "phd" in args.dataset:
        output_file = os.path.join(output_temp_dir, f"{args.subset}_{args.model}.csv")
    else:
        output_file = os.path.join(output_data_dir, f"{args.model}.csv")
    # Output df
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


        ### Step 1.2: Prepare the image path and the prompt
        image_path = os.path.join(image_dir, row[col_image])
        prompt_dir = "../../prompts"
        if not args.prompt_file:
            prompt = row[col_prompt]
        elif "phd" in args.dataset:
            # Special case where context is needed, and thus template is needed
            prompt = open(os.path.join(prompt_dir, args.prompt_file), "r").read().strip()
            if type(row["context"]) is str:
                prompt = prompt.replace("{context}", row["context"])
            else:
                prompt = prompt.replace("{context}", "N/A")
            prompt = prompt.replace("{question}", row[col_prompt])
        else:
            prompt = open(os.path.join(prompt_dir, args.prompt_file), "r").read().strip()
            prompt = prompt.replace("{original_prompt}", row[col_prompt])
        ### Step 1.2.1: Special case for PhD, as it is using both train and val of coco...
        if "phd" in args.dataset:
            if not "ccs" in args.subset and not os.path.exists(image_path):
                image_path = image_path.replace("train", "val")
                row[col_image] = row[col_image].replace("train", "val")


        ### Step 1.3: Infer and process the response
        response = vlm.infer(
                        prompt=prompt, 
                        image_path=image_path,
                        max_new_tokens=512,
                        temperature=args.temperature,
                    )


        ### Step 1.4: Concatenate to the end of df_output
        processed_response = process_response(response, args.dataset)
        df_output = concatenate_response(response, row, df_output, col_image, processed_response=processed_response)


        ### Step 1.5: Write to the output path
        with open(output_file, "w") as f:
            df_output.to_csv(f, index=False)

    

if __name__ == "__main__":
    main(FLAGS)