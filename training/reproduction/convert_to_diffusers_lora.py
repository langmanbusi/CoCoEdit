import torch
from safetensors.torch import load_file, save_file
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen Image Edit with configurable ckpt.")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Checkpoint id or number used to construct output and LoRA paths, e.g., 560",
    )

def main():
    args = parse_args()
    ckpt = args.ckpt
    steps = ['']
    lora_path = "/home/notebook/data/group/wyh/UniWorld-V2/ckpts/nft_qwen_image_edit_exp_nft_qwen32_pixel_cfg4_0_2025.10.31_00.12.42/checkpoints/checkpoint-840/lora"

    base_dir = lora_path

    state_dict = load_file(f"{base_dir}/adapter_model.safetensors")

    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key.replace("base_model.model", "transformer")
        new_state_dict[new_key] = value


    save_file(new_state_dict, f"{base_dir}/adapter_model_converted.safetensors")
