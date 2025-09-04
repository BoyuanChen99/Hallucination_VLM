# Hallucination_VLM
This repo stores the major code and results for the paper "..."






## Getting Datasets
The following benchmarks/datasets are used in this project. Each section also provides a brief instruction on how to download the datasets. We put ALL dataset folders in a directory \textbf{on the top level}, named `data`. 

FYI, the datasets/benchmarks of CV are relatively messy and under-organized compared to other areas in ML. One reason is that it has a relatively long history of research even before ML, so a lot of datasets we still use nowadays were developed a long while ago. Consequently, it takes a decent amount of time to get everything in shape, and thus the purpose of this documentation is to help you save this unnecessarily torturing experience :) 

MSCOCO and COCO are the same thing. In research we usually write the former, emphasizing that the dataset comes from Microsoft, while the latter is more commonly used in codebases/filenames. The rest of the tutorial will use MSCOCO.

We strongly recommend that you start by creating a data folder in a local disk with enough space, and download all the datasets there. The following instructions will assume that you have created a folder named `data` in your home directory.





### [PhD](https://github.com/jiazhen-code/PhD) (CVPR'25)

By far the biggest benchmark dataset on VLM hallucination, including 15,398 distinct images 102,564 VQA triplets. The questions are also very challenging. 







### POPE ([MSCOCO](https://cocodataset.org/#download), [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html), [A-OKVQA](https://github.com/allenai/aokvqa)) (EMNLP'23)

Polling-based Object Probing Evaluation collects images from the following classical datasets for research in computer vision (CV): MSCOCO, A-OKVQA, and GQA. 

For each dataset, they collected images using three negative sample settings: random, popular, and adversarial. Each collection comprises 500 images alongside 6 questions per image. Each question is about the existence of an object (ie. "Is there a snowboard in the image?"), requiring a one-word Yes/No answer. The evaluation metrics include Accuracy, Precision, Recall, and F1 score. 

Note that this dataset is quite old, so several questions are very tricky (fixates on a very small area in the image), or have ambiguous definitions. 

First, download MSCOCO_val2014 and GQA (make sure to click the box of the right-most choice, and download ALL images of 73.9G). A-OKVQA is a subset of MSCOCO, so there is no need to download it separately. 
For your convenience, below is the command to download MSCOCO_val2014:
```bash
wget http://images.cocodataset.org/zips/val2014.zip
```





### CHAIR (MSCOCO) (EMNLP'18)
Caption Hallucination Assessment with Image Relevance (CHAIR) is a benchmark that assesses
caption accuracy by comparing mentioned objects against ground-truth labels, with hallucinations defined as objects present in captions but absent from ground truth. Following established protocol, we evaluate on 500 randomly sampled images from MSCOCO_val2014 set, using the prompt "Please help me describe the image in
detail" with maximum generation length of 512 tokens.

To collect 500 random questions from MSCOCO_val2014, you can use the following command, after setting your own seed:
```bash
python chair_extract.py \ 
    --coco_dir [Your local directory storing MSCOCO_val2014] \
    --chair_dir [Your local directory to store the CHAIR dataset for output] \
    --seed [Your seed]
```



### HaloQuest
The work is published via ECCV 2024 by Google. It uses self-generated original images from Midjourney to ask trickier questions. To prepare this dataset, clone their repo and follow their instructions. 

Note that some images in this dataset are not downloable. The official team did not push updates to the repo to fix this issue.

[paper](https://dl.acm.org/doi/10.1007/978-3-031-72980-5_17), [code](https://github.com/google/haloquest)







## Running Models
This work implements the most-often experimented VLMs: LLaVA-v1.5, LLaVA-NEXT (aka. LLaVA-v1.6), InstructBLIP, MiniGPT-4. We also implemented two latest sota VLMs: Qwen-2.5-VL, InternVL-3. 

While huggingface provides a lot of VLMs, it does not include the full-precision versions of LLaVA-v1.5 and LLaVA-NEXT. For example, the huggingface model llava-v1.5-hf generates sligtly different output from LLaVA-v1.5 in the original repo. For research-level rigor, we forked the original repo and implemented the full-precision versions of [LLaVA-v1.5](https://github.com/BoyuanChen99/LLaVA) and LLaVA-NEXT. Find our open-sourced code for inference. 
