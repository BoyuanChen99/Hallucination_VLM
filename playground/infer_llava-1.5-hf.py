"""
An inference script for Llava-v1.5-7B-hf. Note that while this model has the same level of performance as the original code, it is not a formal implementation. For research-level strictness, please refer to the official implementation at https://github.com/haotian-liu/LLaVA. 
"""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


def main():
    # Step 0: Load the LLaVA-v1.5 model in half-precision
    model_name = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=torch.float16, 
                device_map="auto"
            )
    processor = AutoProcessor.from_pretrained(model_name)

    ### Step 1: Initialize the input conversation as an array of dictionaries
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]

    ### Step 2: Process the conversation with AutoProcessor
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, torch.float16)

    ### Step 3: Generate and print output
    generate_ids = model.generate(**inputs, max_new_tokens=30)
    output = processor.batch_decode(generate_ids, skip_special_tokens=True)
    print(output)
    print(output[0])


if __name__ == "__main__":
    main()