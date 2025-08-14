import pandas as pd
import os
import sys
import argparse

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.abspath(os.path.join(root_dir, 'src')))
from vlms import init_vlm


args = argparse.ArgumentParser(description="Run InternVL3 for HaloQuest evaluation")
# Experiment Group and Category
args.add_argument("--model", type=str, default="InternVL3", help="Model name")
args.add_argument("--dataset", type=str, default="haloquest", help="The dataset to run test on")
# Hyperparameters
args.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
args.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
args.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling")
args.add_argument("--seed", type=int, default=42, help="Random seed")
FLAGS = args.parse_args()


def main(args):
    ### Step 0: Initialize the vlm and the dataset file
    vlm = init_vlm(args.model)
    dataset = args.dataset
    

if __name__ == "__main__":
    main(FLAGS)