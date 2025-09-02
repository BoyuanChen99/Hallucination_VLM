import argparse
import os
from PIL import Image
from gui import ImageViewer
import pandas as pd


def parse_args():
    argparser = argparse.ArgumentParser(description="Run inference with the model.")
    ### VLM and dataset
    argparser.add_argument("--model_name", type=str, default="InternVL3", help="Name of the model to use for inference.")
    argparser.add_argument("--benchmark", type=str, default="chair", help="The benchmark of the experiment. eg: pope, chair, etc.")
    ### Hoper parameters
    argparser.add_argument("--temperature", type=float, default="1.0", help="The temperature used for inference.")
    ### Fixed directories
    argparser.add_argument("--data_folder", type=str, default="../../../data", help="Path to the data folder containing POPE dataset.")
    argparser.add_argument("--subfolder", type=str, default="val2014", help="Only relevant for coco subset.")
    argparser.add_argument("--output_folder", type=str, default="../../results", help="Folder to save the results.")
    return argparser.parse_args()



def main(args):
    ### Step 0: Read the output file, and define the image folder. There is no answers file for CHAIR benchmark. Please run the eval code for CHAIR scores. 
    output_file = os.path.join(args.output_folder, args.benchmark, f"{args.model_name}.csv")
    if not os.path.exists(output_file):
        print(f"Output file {output_file} does not exist. Please run the inference script first.")
        return
    df_results = pd.read_csv(output_file)
    image_dir = os.path.join(args.data_folder, "coco", args.subfolder)
    model_id = args.model_name


    ### Step 1: Loop through the images in the GUI
    images_info = []
    for idx, row in df_results.iterrows():
        question_id = row["idx_question"]
        prompt = row["text"]
        model_output = row["response"].strip().replace("\\n", "\n")
        image_file = row["image"]
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
            "image": image,
            "answer": ""
        })

    if images_info:
        app = ImageViewer(images_info, title=f"CHAIR {args.model_name}, t={args.temperature}", benchmark=args.benchmark)
        app.mainloop()
    else:
        print("No images to display.")



if __name__ == "__main__":
    args = parse_args()
    main(args)
