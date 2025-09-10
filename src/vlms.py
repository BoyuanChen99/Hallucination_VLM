import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM


default_response = "[No response generated]"

### ========== Abstract Model Class ===========
class VLM:
    def __init__(self):
        pass

    def infer(self, *args):
        raise NotImplementedError



# ===== Ovis2-8/16B =====
# This code is adapted from the official team's code on huggingface: https://huggingface.co/AIDC-AI/Ovis2-8B
class Ovis2(VLM):
    def __init__(self, model="AIDC-AI/Ovis2-8B", torch_dtype=torch.bfloat16):
        self.path = model
        self.model = AutoModelForCausalLM.from_pretrained(
                        self.path,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        device_map="auto"
                    ).eval()
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.device = self.model.device
    
    def infer(self, prompt, image_path=None, video=None, max_new_tokens=512, temperature=0.0, max_partition=12):
        response = default_response
        # 1-image 1-round
        if image_path is not None and video is None:
            image = Image.open(image_path).convert('RGB')
            images = [image]
            query = f'<image>\n{prompt}'
            prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, images, max_partition=max_partition)
            attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=self.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=self.device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)
            pixel_values = [pixel_values]
            with torch.inference_mode():
                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    top_p=None,
                    top_k=None,
                    temperature=temperature if temperature > 0 else None,
                    repetition_penalty=None,
                    eos_token_id=self.model.generation_config.eos_token_id,
                    pad_token_id=self.text_tokenizer.pad_token_id,
                    use_cache=True
                )
                output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
                response = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return response



# ===== InternVL3-8/14B =====
# This code is adapted from the official team's code on huggingface: https://huggingface.co/OpenGVLab/InternVL3-8B. Note that for this model, "max_num" means "max_partition".
class InternVL3(VLM):
    ### ========== Initialization ========== ###
    def __init__(self, model="OpenGVLab/InternVL3-8B", torch_dtype=torch.bfloat16, load_in_8bit=False, use_flash_attn=True):
        # Note that the image dimension must be fixed. The original code uses Imagenet settings by default. 
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.path = model
        # Detect how many gpus are available. If there are more than 1, we will split the model evenly.
        if torch.cuda.device_count() > 1:
            self.device_map = self.split_model(self.path)
            self.model = AutoModel.from_pretrained(
                            self.path,
                            torch_dtype=torch_dtype,
                            load_in_8bit=load_in_8bit,
                            use_flash_attn=use_flash_attn,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            device_map=self.device_map
                        ).eval()
        else:
            self.model = AutoModel.from_pretrained(
                            self.path,
                            torch_dtype=torch_dtype,
                            load_in_8bit=load_in_8bit,
                            use_flash_attn=use_flash_attn,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True
                        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
                        self.path, 
                        trust_remote_code=True, 
                        use_fast=False
                    )
        self.device = self.model.device

    ### ========== Helper functions ========== ###
    def build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    ### A helper function used for any AutoModel that runs on multiple gpus. The reason for writing the code this way is to avoid errors that occur during multi-GPU inference due to tensors not being on the same device. By ensuring that the first and last layers of the large language model (LLM) are on the same device, we prevent such errors. The returned output is a dictionary that maps layer names to device IDs.
    def split_model(self, model_name):
        device_map = {}
        world_size = torch.cuda.device_count()
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        num_layers = config.llm_config.num_hidden_layers
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.model.rotary_emb'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
        return device_map

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        # Calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        # Find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)
        # Calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        # Resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # Split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    ### ========== Video-related functions ========== ###
    # Video multi-round conversation
    def get_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices

    def load_video(self, video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=input_size)
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def infer(self, prompt, image_path=None, video=None, max_new_tokens=512, temperature=0.0, multi_rounds=False):
        response = default_response
        # 1-image 1-round
        if video is None and image_path is not None and not multi_rounds:
            pixel_values = self.load_image(image_path, max_num=12).to(torch.bfloat16).to(self.device)
            question = f"<image>\n{prompt}"
            if temperature > 0:
                generation_config = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
            else:
                generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
            response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        return response


### ========== 
def init_vlm(model_name):
    if "internvl3" in model_name.lower():
        return InternVL3(model=model_name)
    elif "ovis2" in model_name.lower():
        return Ovis2(model=model_name)
    raise ValueError(f"Unknown model name: {model_name}")