import os
import json
import argparse
from PIL import Image
import torch
from tqdm import tqdm

from diffusers import QwenImageEditPlusPipeline
from safetensors.torch import load_file, save_file

def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen Image Edit with configurable ckpt.")
    parser.add_argument(
        "--json_path",
        type=str,
        default="ImgEdit/bench-data/singleturn.json",
        help="Path to the JSON metafile.",
    )
    parser.add_argument(
        "--output_root_tpl",
        type=str,
        default="results/qwen-image-edit2509-40step-cocoedit",
        help="Template for output root.",
    )
    parser.add_argument(
        "--lora_path_tpl",
        type=str,
        default="qwen-cocoedit-lora/lora",
        help="Template for LoRA path.",
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Disable LoRA even if lora_path_tpl is provided.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    json_path = args.json_path
    output_root = args.output_root_tpl
    lora_path = None
    if not args.no_lora and args.lora_path_tpl:
        lora_path = args.lora_path_tpl

    print(f"output_root: {output_root}")
    print(f"lora_path: {lora_path}")

    pipeline = QwenImageEditPlusPipeline.from_pretrained("ckpt/Qwen-Image-Edit-2509")
    print("pipeline loaded")
    pipeline.to(torch.bfloat16)
    pipeline.to("cuda")
    if lora_path is not None:
        print(f"Lora path provided: {lora_path}")
        converted_safetensor_path = f'{lora_path}/adapter_model_converted.safetensors'
        if not os.path.exists(converted_safetensor_path):
            print('no converted lora, converting...')
            state_dict = load_file(f"{lora_path}/adapter_model.safetensors")
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("base_model.model", "transformer")
                new_state_dict[new_key] = value
            save_file(new_state_dict, f"{lora_path}/adapter_model_converted.safetensors")

        pipeline.load_lora_weights(
            lora_path,
            weight_name="adapter_model_converted.safetensors",
            # weight_name="adapter_model.safetensors",
            adapter_name="lora",
        )
        pipeline.set_adapters(["lora"], adapter_weights=[1])
        print("Lora loaded")
    pipeline.set_progress_bar_config(disable=None)

    with open(json_path, "r") as f:
        data = json.load(f)

    for key, sample in tqdm(data.items()):
        ref_image_path = "ImgEdit/bench-data/" + sample["id"]
        prompt = sample["prompt"]
        save_path = os.path.join(output_root, f'{key}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image = Image.open(ref_image_path).convert("RGB")

        inputs = {
            "image": image,
            "prompt": prompt,
            "generator": torch.manual_seed(0),
            "true_cfg_scale": 4.0,
            "guidance_scale": 1.0,
            "negative_prompt": " ",
            "num_inference_steps": 40,
        }

        with torch.inference_mode():
            output = pipeline(**inputs)
            output_image = output.images[0]
            output_image.save(save_path)
            print("image saved at", save_path)

    print("All images processed.")

if __name__ == "__main__":
    main()
