# Hallucination_VLM
This repo stores the major code and results for the paper "..."

## Getting Datasets
The following benchmarks/datasets are used in this project. Each section also provides a brief instruction on how to download the datasets.

A few clarifications before you start:

The datasets of CV are very messy and ill-managed, compared to other areas in ML. One reason is that it has a relatively long history of research even before ML. Consequently, it takes a decent amount of time to get everything in shape, and thus the purpose of this tutorial is to help you save this unnecessarily torturing experience :) 

MSCOCO and COCO are the same thing. In research we usually write the former, emphasizing that the dataset comes from Microsoft, while the latter is more commonly used in codebases/filenames. The rest of the tutorial will use MSCOCO.

We strongly recommend that you start by creating a data folder in a local disk with enough space, and download all the datasets there. The following instructions will assume that you have created a folder named `data` in your home directory.

### POPE ([MSCOCO](https://cocodataset.org/#download), [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html), A-OKVQA)

Polling-based Object Probing Evaluation collects images from the following classical datasets for research in computer vision (CV): MSCOCO, A-OKVQA, and GQA. 
For each dataset, they collected images using three negative sample settings: random, popular, and adversarial. Each collection comprises 500 images alongside 6 questions per image. Each question is about the existence of an object (ie. "Is there a snowboard in the image?"), requiring a one-word Yes/No answer. The evaluation metrics include Accuracy, Precision, Recall, and F1 score. Following previous works, we run experiments and exhibit results on all questions for each collection from each dataset.

First, download MSCOCO_val2014 and GQA (make sure to click the box of the right-most choice, and download ALL images of 73.9G). A-OKVQA is a subset of MSCOCO, so there is no need to download it separately. 
For your convenience, below is the command to download MSCOCO_val2014:
```bash
wget http://images.cocodataset.org/zips/val2014.zip
```



### CHAIR (MSCOCO)

Caption Hallucination Assessment with Image Relevance (CHAIR) is a benchmark that assesses
caption accuracy by comparing mentioned objects against ground-truth labels, with hallucinations defined as objects present in captions but absent from ground truth. Following established protocol, we evaluate on 500 randomly sampled images from MSCOCO_val2014 set, using the prompt "Please help me describe the image in
detail" with maximum generation length of 512 tokens.




LLaVA-Bench

MME

## Running Models
This work implements the most-often experimented VLMs: LLaVA-v1.5, LLaVA-NEXT (aka. LLaVA-v1.6), InstructBLIP, MiniGPT-4. We also implemented two latest sota VLMs: Qwen-2.5-VL, InternVL-3. 

While huggingface provides a lot of VLMs, it does not include the full-precision versions of LLaVA-v1.5 and LLaVA-NEXT. For example, the huggingface model llava-v1.5-hf generates sligtly different output from LLaVA-v1.5 in the original repo. For research-level rigor, we forked the original repo and implemented the full-precision versions of [LLaVA-v1.5](https://github.com/BoyuanChen99/LLaVA) and LLaVA-NEXT. Find our open-sourced code for inference. 
