"""
https://huggingface.co/AIDC-AI/Ovis2-8B
"""

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.abspath(os.path.join(root_dir, 'src')))

from vlms import init_vlm
from utils import disable_huggingface_warnings

def main():
    ### Step 0: Initialize the VLM and the data
    # Note that while the size of the model itself isn't large, it takes a lot of memory to process image data.
    # If you set `load_in_8bit=True`, you will need two 80GB GPUs.
    # If you set `load_in_8bit=False`, you will need about 14G of VRAM to process an image.
    disable_huggingface_warnings()
    model = init_vlm("AIDC-AI/Ovis2-8B")

    # single-image input
    image_path = "../../data/coco/val2014/COCO_val2014_000000000073.jpg"
    text = 'Describe the image.'
    output = model.infer(prompt=text, image_path=image_path, temperature=0.0, max_partition=12)
    print(output)

    ## cot-style input
    # cot_suffix = "Provide a step-by-step solution to the problem, and conclude with 'the answer is' followed by the final solution."
    # image_path = '/data/images/example_1.jpg'
    # images = [Image.open(image_path)]
    # max_partition = 9
    # text = "What's the area of the shape?"
    # query = f'<image>\n{text}\n{cot_suffix}'

    ## multiple-images input
    # image_paths = [
    #     '/data/images/example_1.jpg',
    #     '/data/images/example_2.jpg',
    #     '/data/images/example_3.jpg'
    # ]
    # images = [Image.open(image_path) for image_path in image_paths]
    # max_partition = 4
    # text = 'Describe each image.'
    # query = '\n'.join([f'Image {i+1}: <image>' for i in range(len(images))]) + '\n' + text

    ## video input (require `pip install moviepy==1.0.3`)
    # from moviepy.editor import VideoFileClip
    # video_path = '/data/videos/example_1.mp4'
    # num_frames = 12
    # max_partition = 1
    # text = 'Describe the video.'
    # with VideoFileClip(video_path) as clip:
    #     total_frames = int(clip.fps * clip.duration)
    #     if total_frames <= num_frames:
    #         sampled_indices = range(total_frames)
    #     else:
    #         stride = total_frames / num_frames
    #         sampled_indices = [min(total_frames - 1, int((stride * i + stride * (i + 1)) / 2)) for i in range(num_frames)]
    #     frames = [clip.get_frame(index / clip.fps) for index in sampled_indices]
    #     frames = [Image.fromarray(frame, mode='RGB') for frame in frames]
    # images = frames
    # query = '\n'.join(['<image>'] * len(images)) + '\n' + text

    ## text-only input
    # images = []
    # max_partition = None
    # text = 'Hello'
    # query = text


if __name__ == "__main__":
    main()