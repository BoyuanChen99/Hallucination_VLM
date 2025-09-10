"""
A standard inference script for InternVL-3-8B. This model is only supported via custom code.
The code is adapted from the official huggingface page: https://huggingface.co/OpenGVLab/InternVL3-8B, and is currently stored in `src/vlms.py`.
This model is the TOP1 of huggingface vlm leaderboard by August 12th, 2025. 
"""

import os
import sys
import torch
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.abspath(os.path.join(root_dir, 'src')))

from vlms import InternVL3
from utils import disable_huggingface_warnings


def main():
    ### Step 0: Initialize the VLM and the data
    # Note that while the size of the model itself isn't large, it takes a lot of memory to process image data.
    # If you set `load_in_8bit=True`, you will need two 80GB GPUs.
    # If you set `load_in_8bit=False`, you will need about 14G of VRAM to process an image.
    disable_huggingface_warnings()
    internvlm = InternVL3(torch_dtype=torch.bfloat16, load_in_8bit=True, use_flash_attn=True)
    image_1_path = "../../data/coco/val2014/COCO_val2014_000000000073.jpg"
    image_2_path = "../../data/coco/val2014/COCO_val2014_000000000074.jpg"


    ### Step 1: Set the max number of tiles in `max_num`
    pixel_values_1 = internvlm.load_image(image_1_path, max_num=12).to(torch.bfloat16).to(internvlm.device)
    generation_config = dict(max_new_tokens=512, do_sample=True)


    ### Step 2: pure-text conversation (纯文本对话), note that the history is preserved
    print(f"\n\n\n========== Step 2: Pure-text conversation ==========\n")
    question = 'Hello, who are you?'
    response, history = internvlm.model.chat(
                            internvlm.tokenizer, 
                            None, 
                            question, 
                            generation_config, 
                            history=None, 
                            return_history=True
                        )
    print(f'User: {question}\nAssistant: {response}')
    question = 'Can you tell me a story?'
    response, history = internvlm.model.chat(
                            internvlm.tokenizer, 
                            None, 
                            question, 
                            generation_config, 
                            history=history, 
                            return_history=True
                        )
    print(f'User: {question}\nAssistant: {response}')
    input()


    ### Step 3: single-image single-round conversation (单图单轮对话)
    print(f"\n\n\n========== Step 3: Single-image single-round conversation ==========\n")
    question = '<image>\nPlease describe the image shortly.'
    response = internvlm.model.chat(internvlm.tokenizer, pixel_values_1, question, generation_config)
    print(f'User: {question}\nAssistant: {response}')
    input()


    ### Step 4: single-image multi-round conversation (单图多轮对话)
    print(f"\n\n\n========== Step 4: Single-image multi-round conversation ==========\n")
    question = '<image>\nPlease describe the image in detail.'
    response, history = internvlm.model.chat(internvlm.tokenizer, pixel_values_1, question, generation_config, history=None, return_history=True)
    print(f'User: {question}\nAssistant: {response}')
    question = 'Please write a poem according to the image.'
    response, history = internvlm.model.chat(internvlm.tokenizer, pixel_values_1, question, generation_config, history=history, return_history=True)
    print(f'User: {question}\nAssistant: {response}')
    input()

    
    ### Step 5: multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
    print(f"\n\n\n========== Step 5: Multi-image multi-round conversation, combined images ==========\n")
    pixel_values1 = internvlm.load_image(image_1_path, max_num=12).to(torch.bfloat16).cuda()
    pixel_values2 = internvlm.load_image(image_2_path, max_num=12).to(torch.bfloat16).cuda()
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

    question = '<image>\nDescribe the two images in detail.'
    response, history = internvlm.model.chat(internvlm.tokenizer, pixel_values, question, generation_config,
                                history=None, return_history=True)
    print(f'User: {question}\nAssistant: {response}')

    question = 'What are the similarities and differences between these two images.'
    response, history = internvlm.model.chat(internvlm.tokenizer, pixel_values, question, generation_config,
                                history=history, return_history=True)
    print(f'User: {question}\nAssistant: {response}')
    input()


    ### Step 6: multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
    print(f"\n\n\n========== Step 6: Multi-image multi-round conversation, separate images ==========\n")
    pixel_values1 = internvlm.load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
    pixel_values2 = internvlm.load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

    question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
    response, history = internvlm.model.chat(internvlm.tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list,
                                history=None, return_history=True)
    print(f'User: {question}\nAssistant: {response}')

    question = 'What are the similarities and differences between these two images.'
    response, history = internvlm.model.chat(internvlm.tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list,
                                history=history, return_history=True)
    print(f'User: {question}\nAssistant: {response}')
    input()


    ### Step 7: batch inference, single image per sample
    pixel_values1 = internvlm.load_image(image_1_path, max_num=12).to(torch.bfloat16).cuda()
    pixel_values2 = internvlm.load_image(image_2_path, max_num=12).to(torch.bfloat16).cuda()
    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

    questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
    responses = internvlm.model.batch_chat(internvlm.tokenizer, pixel_values,
                                num_patches_list=num_patches_list,
                                questions=questions,
                                generation_config=generation_config)
    for question, response in zip(questions, responses):
        print(f'User: {question}\nAssistant: {response}')
    input()


    ### Step 8: Video processing
    video_path = './examples/red-panda.mp4'
    pixel_values, num_patches_list = internvlm.load_video(video_path, num_segments=8, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    question = video_prefix + 'What is the red panda doing?'
    # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
    response, history = internvlm.model.chat(internvlm.tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list, history=None, return_history=True)
    print(f'User: {question}\nAssistant: {response}')
    question = 'Describe this video in detail.'
    response, history = internvlm.model.chat(internvlm.tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list, history=history, return_history=True)
    print(f'User: {question}\nAssistant: {response}')



if __name__ == '__main__':
    main()