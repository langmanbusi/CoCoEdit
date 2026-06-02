# reward-server

This repository contains a Flask + Ray reward inference service, implemented as a QwenServer deployment. The server starts from `qwenserver.py`, and the startup script is provided in `app_qwenserver.sh`.

## Acknowledgements

This implementation is based on the EditScore framework from:
```
@article{luo2025editscore,
  title={EditScore: Unlocking Online RL for Image Editing via High-Fidelity Reward Modeling},
  author={Xin Luo and Jiahao Wang and Chenyuan Wu and Shitao Xiao and Xiyan Jiang and Defu Lian and Jiajun Zhang and Dong Liu and Zheng Liu},
  journal={arXiv preprint arXiv:2509.23909},
  year={2025}
}
```

We thank the authors for their research and the foundation it provides for reward modeling in image editing.

## Structure

- `qwenserver.py`: Flask + Ray service entry point. Accepts pickle-encoded requests, dispatches work to Ray actors, and returns results.
- `reward_server/editscore_eval.py`: EditScore model loader and inference wrapper.
- `app_qwenserver.sh`: Startup script that activates the conda environment and runs `qwenserver.py`.

## Setup

Recommended usage with an existing `qwen` conda environment, or create one as follows:

```bash
conda create -n qwen python=3.10
conda activate qwen
pip install torch torchvision flask ray pillow numpy
```

If you use GPU, install the matching `torch`/`torchvision` build for your CUDA version.

## Start the Service

Run the startup script:

```bash
bash app_qwenserver.sh
```

Alternatively, start manually:

```bash
conda activate qwen
python qwenserver.py
```

## Environment Variables

Optional environment variables to control server behavior:

- `NUM_GPUS`: total available GPUs, default `8`
- `NUM_TP`: GPUs per Ray actor, default `1`
- `SERVER_BATCH`: server-side batch size, default `16`
- `MAX_CONTENT_LENGTH_MB`: Flask max request size, default `64`
- `PORT`: Flask listen port, default `18086`

## API

### Health Check

- `GET /health`
- Returns `ok` when the service is ready.

### Inference Endpoint

- `POST /`
- Request body is a pickle-encoded Python object with the following structure:

```python
{
  "images": [bytes, ...],
  "output_images": [bytes, ...],
  "prompts": [str, ...]
}
```

- Response is a pickle-encoded object with an `outputs` field:

```python
{"outputs": (scores, extra, results)}
```

The function `load_editscore_0_5()` in `reward_server/editscore_eval.py` returns an inference wrapper that supports PIL images or tensors and outputs normalized scores in `[0, 1]`.

## `reward_server/editscore_eval.py` Details

This implementation:

- uses the `qwen25vl_vllm` backbone
- converts input images to RGB
- calls `scorer.evaluate([ref_image, image], prompt)` for each sample
- divides the raw score by `5` to normalize it
- returns `0.5` for invalid or non-finite values

If you need to customize the model path, modify the `model_path` variable.
