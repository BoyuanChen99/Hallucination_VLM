import argparse
import os
import json
from PIL import Image
from gui import ImageViewer
import pandas as pd


def parse_args():
    argparser = argparse.ArgumentParser(description="Run inference with the model.")
    ### VLM and dataset
    argparser.add_argument("--model_name", type=str, default="InternVL3", help="Name of the model to use for inference.")
    argparser.add_argument("--benchmark", type=str, default="pope", help="The benchmark of the experiment. eg: pope, chair, etc.")
    ### Hoper parameters
    argparser.add_argument("--temperature", type=float, default="0.0", help="The temperature used for inference.")
    argparser.add_argument("--subset", type=str, default="coco", help="Subset of the POPE dataset to use. The three options are: coco, aokvqa, and gqa.")
    argparser.add_argument("--subsplit", type=str, default="popular", help="Subsplit of the POPE dataset to use.")
    ### Fixed directories
    argparser.add_argument("--data_folder", type=str, default="../../../data", help="Path to the data folder containing POPE dataset.")
    argparser.add_argument("--subfolder", type=str, default="val2014", help="Only relevant for coco subset.")
    argparser.add_argument("--output_folder", type=str, default="../../results", help="Folder to save the results.")
    return argparser.parse_args()



def main(args):
    ### Step 0.1: Define, and read the output file
    if args.temperature <= 0:
        temperature = "greedy"
    else:
        temperature = str(args.temperature)
        if temperature.endswith(".0"):
            temperature = "temp"+temperature[:-2]
    # Load output
    output_file_jsonl = os.path.join(args.output_folder, args.benchmark, args.model_name, temperature, f"{args.subset}_pope_{args.subsplit}_{args.model_name}.jsonl")
    output_file_csv = os.path.join(args.output_folder, args.benchmark, args.model_name, temperature, f"{args.subset}_{args.model_name}_{args.subsplit}.csv")
    print(output_file_csv)
    if os.path.exists(output_file_jsonl):
        with open(output_file_jsonl, "r") as f:
            results = [json.loads(line) for line in f.readlines()]
        ### Step 0.2: Define, and read the answers file
        answers_file = os.path.join(args.data_folder, args.benchmark, args.subset, f"{args.subset}_{args.benchmark.lower()}_{args.subsplit}.json")
        if not os.path.exists(answers_file):
            print(f"Answers file {answers_file} does not exist. Please check the path.")
            return
        with open(answers_file, "r") as f:
            answers = [json.loads(line) for line in f]
    elif os.path.exists(output_file_csv):
        with open(output_file_csv, "r") as f:
            df = pd.read_csv(f)
            answers = df.to_dict(orient="records")
            df = df.rename(columns={"text": "prompt"})
            df = df.rename(columns={"response": "text"})
            results = df.to_dict(orient="records")

    ### Step 0.3: Define image folder
    args.subset = args.subset.lower()
    if "gqa" in args.subset:
        image_dir = os.path.join(args.data_folder, args.subset, "allImages", "images")
    elif "coco" in args.subset or "aokvqa" in args.subset:
        image_dir = os.path.join(args.data_folder, args.subset, args.subfolder)


    ### Step 1: Loop through the images in the GUI
    images_info = []
    for result in results:
        question_id = result["question_id"]
        prompt = result["prompt"]
        model_output = result["text"].lower().strip()
        model_id = result.get("model_id", args.model_name)
        image_file = result["image"]
        answer = next(
            (a["label"] for a in answers 
            if a["question_id"] == question_id and a["text"] == prompt and a["image"] == image_file),
            "NOT FOUND"
        )
        image_path = os.path.join(image_dir, image_file)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        image = Image.open(image_path).convert("RGB")
        images_info.append({
            "question_id": question_id,
            "prompt": prompt,
            "model_output": model_output,
            "model_id": model_id,
            "image_file": image_file,
            "answer": answer,
            "image": image,
        })

    if images_info:
        app = ImageViewer(images_info, title=f"POPE {args.subset} {args.subsplit} {args.model_name}, t={args.temperature}")
        app.mainloop()
    else:
        print("No images to display.")



if __name__ == "__main__":
    args = parse_args()
    main(args)
