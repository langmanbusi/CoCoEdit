# CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning

[![arXiv](https://img.shields.io/badge/arXiv-CoCoEdit-b31b1b.svg)](https://arxiv.org/abs/2602.14068) 
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://github.com/langmanbusi/CoCoEdit)
[![Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://github.com/langmanbusi/CoCoEdit)
[![Project](https://img.shields.io/badge/Project-Page-blue)](https://github.com/langmanbusi/CoCoEdit)

Yuhui Wu<sup>1,2</sup>, Chenxi Xie<sup>1,2</sup>, Ruibin Li<sup>1</sup>, Liyi Chen<sup>1</sup>, Qiaosi Yi<sup>1,2</sup>, Lei Zhang<sup>1,2</sup>*

(*Corresponding Author)

<sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>OPPO Research Institute

### Abstract

<details><summary>Click for the full text</summary>
Image editing has achieved impressive results with the development of large-scale generative models. However, existing models mainly focus on the editing effects of intended objects and regions, often leading to unwanted changes in unintended regions. 
We present a post-training framework for Content-Consistent Editing (CoCoEdit) via region regularized reinforcement learning. 
We first augment existing editing datasets with refined instructions and masks, from which 40K diverse and high quality samples are curated as training set. We then introduce a pixel-level similarity reward to complement MLLM-based rewards, enabling models to ensure both editing quality and content consistency during the editing process. To overcome the spatial-agnostic nature of the rewards, we propose a region-based regularizer, aiming to preserve non-edited regions for high-reward samples while encouraging editing effects for low-reward samples. 
For evaluation, we annotate editing masks for GEdit-Bench and ImgEdit-Bench, introducing pixel-level similarity metrics to measure content consistency and editing quality.
Applying CoCoEdit to Qwen-Image-Edit and FLUX-Kontext, we achieve not only competitive editing scores with state-of-the-art models, but also significantly better content consistency, measured by PSNR/SSIM metrics and human subjective ratings.
</details>

### Updates
- 


### TODO 
- [ ] Release the pretrained model.
- [ ] Update the code for inference.
- [ ] Release the dataset.
- [ ] Update the code for training.


## Citation

If you find this work helpful, please consider citing:

```
@article{wu2026cocoedit,
      title={CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning}, 
      author={Yuhui Wu and Chenxi Xie and Ruibin Li and Liyi Chen and Qiaosi Yi and Lei Zhang},
      year={2026},
}
```
