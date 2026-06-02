import torch
from PIL import Image
import numpy as np
from .editscore import EditScore, EditScore_0_5

def load_editscore_0_5():
    # Load the EditScore model. It will be downloaded automatically.
    # Replace with the specific model version you want to use.
    model_path = "/home/notebook/data/group/wyh/ckpt/Qwen2.5-VL-32B-Instruct"
    # lora_path = "/home/notebook/data/group/wyh/ckpt/EditScore-7B"

    scorer = EditScore_0_5(
        backbone="qwen25vl_vllm", # set to "qwen25vl_vllm" for faster inference
        model_name_or_path=model_path,
        enable_lora=False,
        # lora_path=lora_path,
        score_range=5,
        num_pass=1, # Increase for better performance via self-ensembling
    )
    print("--Editscore Loaded--")

    @torch.no_grad()
    def _fn(images, ref_images, prompts):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
            ref_images = (ref_images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            ref_images = ref_images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            ref_images = [Image.fromarray(ref_image) for ref_image in ref_images]
        prompts = [prompt for prompt in prompts]
        scores = []
        results = []
        for i in range(len(prompts)):
            result = scorer.evaluate([ref_images[i], images[i]], prompts[i])
            results.append(result)
            # val = result.get("overall", None)
            val = result
            if isinstance(val, (int, float)) and np.isfinite(val):
                scores.append(float(val / 5))
            else:
                scores.append(0.5)
        return scores, {}, results

    return _fn
