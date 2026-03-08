<div align="center">
      
# CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning

[![arXiv](https://img.shields.io/badge/arXiv-CoCoEdit-b31b1b.svg)](https://arxiv.org/abs/2602.14068) 
[![Project](https://img.shields.io/badge/Project-Page-blue)](https://langmanbusi.github.io/CoCo-Edit/)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://huggingface.co/datasets/wyh6666/CoCoEdit-40K)
[![Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/wyh6666/CoCoEdit)


Yuhui Wu<sup>1,2</sup>, Chenxi Xie<sup>1,2</sup>, Ruibin Li<sup>1</sup>, Liyi Chen<sup>1</sup>, Qiaosi Yi<sup>1,2</sup>, Lei Zhang<sup>1,2</sup>*

(*Corresponding Author)

<sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>OPPO Research Institute

> *Curate CoCoEdit-40K for content-consistent training, alleviate the over-edit issue of open-source models through RL, and extend existing benchmarks for consistency evaluation.*
</div>


<div align="center">
    <img src="images/teaser.png" width="900">
</div>

### Abstract

<details><summary>Click for the full text</summary>
Image editing has achieved impressive results with the development of large-scale generative models. However, existing models mainly focus on the editing effects of intended objects and regions, often leading to unwanted changes in unintended regions. 
We present a post-training framework for Content-Consistent Editing (CoCoEdit) via region regularized reinforcement learning. 
We first augment existing editing datasets with refined instructions and masks, from which 40K diverse and high quality samples are curated as training set. We then introduce a pixel-level similarity reward to complement MLLM-based rewards, enabling models to ensure both editing quality and content consistency during the editing process. To overcome the spatial-agnostic nature of the rewards, we propose a region-based regularizer, aiming to preserve non-edited regions for high-reward samples while encouraging editing effects for low-reward samples. 
For evaluation, we annotate editing masks for GEdit-Bench and ImgEdit-Bench, introducing pixel-level similarity metrics to measure content consistency and editing quality.
Applying CoCoEdit to Qwen-Image-Edit and FLUX-Kontext, we achieve not only competitive editing scores with state-of-the-art models, but also significantly better content consistency, measured by PSNR/SSIM metrics and human subjective ratings.
</details>

### TODO 
- [x] Project page. 
- [x] Release arXiv version.
- [x] Release the pretrained model.
- [x] Release the dataset.
- [ ] Update the code for training.


## Inference

### Quick start
Install the latest version of diffusers
```python
pip install git+https://github.com/huggingface/diffusers
```

Built on the official codes of [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit-2509), you can run the following script to inference with our CoCoEdit LoRA [weights](https://huggingface.co/wyh6666/CoCoEdit). 

``` python
import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipeline.to('cuda')

lora_path = 'qwen-cocoedit-lora/lora'
pipeline.load_lora_weights(
    lora_path,
    weight_name="adapter_model_converted.safetensors",
    adapter_name="lora",
)
pipeline.set_adapters(["lora"], adapter_weights=[1])
print("Lora loaded")

pipeline.set_progress_bar_config(disable=None)
image = Image.open("input.png")
prompt = "your prompt here."
inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output.png")
    print("image saved at", os.path.abspath("output.png"))

```

### Benchmark inference

You can also conduct benchmark inference on [ImgEdit-Bench](https://github.com/PKU-YuanGroup/ImgEdit/blob/main/Benchmark/Basic/basic_bench_readme.md) and [GEdit-Bench](https://huggingface.co/datasets/stepfun-ai/GEdit-Bench) using ```test_gedit-bench.py``` and ```test_imgedit-bench.py```.

## CoCoEdit-40K

You can download our CoCoEdit-40K dataset on the huggingface [link](https://huggingface.co/datasets/wyh6666/CoCoEdit-40K) shown above. Due to the limited uploading speed, we split the zips as shown below. You could run ```cat images_part1.zip_* > images_part1.zip``` to restore each zip.

```
CoCoEdit-40K/
    - metadata.jsonl
    - images_part1.zip_aa
    - images_part1.zip_ab
    - images_part1.zip_ac
    - images_part2.zip_aa
    - ...
    - masks_part1.zip
    - ...
```

## Citation

If you find this work helpful, please consider citing:

```
@article{wu2026cocoedit,
  title={CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning},
  author={Wu, Yuhui and Xie, Chenxi and Li, Ruibin and Chen, Liyi and Yi, Qiaosi and Zhang, Lei},
  journal={arXiv preprint arXiv:2602.14068},
  year={2026}
}
```
