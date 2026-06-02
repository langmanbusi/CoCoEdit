from PIL import Image
import io
import os
import numpy as np
import torch
from collections import defaultdict
import random
import requests
from requests.adapters import HTTPAdapter, Retry
from io import BytesIO
import pickle
import math
from flow_grpo.psnr_scorer import MaskedPSNRSSIMScorer, MaskedSSIMCharbonnierScorer
import ipdb


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew / 500, meta

    return _fn

def mllm_score_remote_new(device):
    import os
    import math
    import pickle
    import numpy as np
    import torch
    from io import BytesIO
    from PIL import Image
    import requests
    from requests.adapters import HTTPAdapter, Retry

    # 服务地址：支持 MLLM_SERVER=host:port 或 http(s)://host:port
    base = os.getenv("MLLM_SERVER", "localhost:18086")
    url = base if base.startswith("http") else f"http://{base}"

    # Session + 重试
    sess = requests.Session()
    # 连接失败/瞬时 5xx 的重试；方法不限（allowed_methods=None 等价于全部）
    retries = Retry(
        total=8,
        connect=8,
        read=0,            # 推理耗时长，read 不做自动重试，避免重复计算
        status=5,
        backoff_factor=1.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=None,
        raise_on_status=False,
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))
    sess.mount("https://", HTTPAdapter(max_retries=retries))

    # 超时：区分连接与读取（连接阶段快失败，处理阶段给足时间）
    connect_timeout = 5
    read_timeout = 300

    # 批大小：受服务端 MAX_CONTENT_LENGTH 和模型显存影响
    batch_size = int(os.getenv("MLLM_BATCH", "4"))

    # 可选：启动前做一次健康检查（不报错，只记录）
    try:
        health = sess.get(f"{url}/health", timeout=(3, 3))
        if health.status_code != 200:
            print(f"[warn] health not ready: {health.status_code} {health.text[:200]}")
    except Exception as e:
        print(f"[warn] health check failed: {e}")

    def _to_uint8_nhwc(x):
        if isinstance(x, torch.Tensor):
            arr = (x * 255).round().clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
            if arr.ndim == 4 and arr.shape[1] in (1, 3, 4):  # NCHW -> NHWC
                arr = arr.transpose(0, 2, 3, 1)
            return arr
        arr = np.asarray(x)
        return arr

    def _encode_jpeg(arr):
        # arr: HWC [uint8]
        img = Image.fromarray(arr)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=95, optimize=True)
        return buf.getvalue()

    def _fn(images, ref_images, prompts):
        images_np = _to_uint8_nhwc(images)
        ref_np = _to_uint8_nhwc(ref_images)
        prompts_list = list(prompts)

        assert len(images_np) == len(ref_np) == len(prompts_list), "images/ref_images/prompts 数量需一致"

        N = len(prompts_list)
        num_batches = math.ceil(N / batch_size)

        scores_all = []
        results_all = []

        for b in range(num_batches):
            s = b * batch_size
            e = min((b + 1) * batch_size, N)

            jpeg_images = [_encode_jpeg(images_np[i]) for i in range(s, e)]
            jpeg_ref_images = [_encode_jpeg(ref_np[i]) for i in range(s, e)]
            batch_prompts = prompts_list[s:e]

            payload = pickle.dumps({
                "output_images": jpeg_images,   # 编辑后的图片
                "images": jpeg_ref_images,      # 参考图片
                "prompts": batch_prompts,
            })

            try:
                resp = sess.post(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=(connect_timeout, read_timeout),
                )
            except requests.exceptions.ConnectTimeout:
                # 连接阶段失败：可快速重试或降载
                raise
            except requests.exceptions.ReadTimeout:
                # 处理超时：给出提示或降批/重试
                raise

            # 非 2xx 先尽量打印服务端文本（可能是 traceback）
            if not resp.ok:
                text_preview = resp.text[:500] if resp.headers.get("Content-Type", "").startswith("text") else ""
                raise RuntimeError(f"Server error: {resp.status_code}. {text_preview}")

            # 正常返回为 pickle 字节
            resp_obj = pickle.loads(resp.content)

            outputs = resp_obj.get("outputs")
            if outputs is None:
                batch_scores = []
                batch_results = []
            else:
                # 兼容元组/列表形式
                if isinstance(outputs, (list, tuple)):
                    # 你原代码约定 outputs[0]=scores, outputs[1]=dict, outputs[2]=results
                    batch_scores = list(outputs[0]) if len(outputs) > 0 else []
                    batch_results = list(outputs[2]) if len(outputs) > 2 else []
                else:
                    # 若后端仅返回标量分数数组
                    batch_scores = list(outputs)
                    batch_results = []

            # 清洗分数
            cleaned = []
            for v in batch_scores:
                try:
                    vf = float(v)
                    if np.isfinite(vf):
                        cleaned.append(vf)
                    else:
                        cleaned.append(0.5)
                except Exception:
                    cleaned.append(0.5)

            scores_all.extend(cleaned)
            results_all.extend(batch_results)

        # 返回: scores, extra_info(暂空), results
        return scores_all, {}, results_all

    return _fn

def mllm_score_remote(device):
    # url = "http://10.77.227.218:18085"
    url = f"http://{os.getenv('MLLM_SERVER', 'localhost:18085')}"
    # url = "http://10.77.227.60:12341/mode/logits_non_cot"

    sess = requests.Session()
    retries = Retry(
        total=10, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))
    batch_size = 4

    def _fn(images, ref_images, prompts):
        # 规范化输入为 numpy NHWC uint8
        if isinstance(images, torch.Tensor):
            images_np = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images_np = images_np.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        else:
            images_np = np.asarray(images)  # 期望为 [N,H,W,C] uint8 或可转为此
        if isinstance(ref_images, torch.Tensor):
            ref_np = (ref_images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            ref_np = ref_np.transpose(0, 2, 3, 1)
        else:
            ref_np = np.asarray(ref_images)

        # 统一 prompts 为 list[str]
        prompts_list = list(prompts)

        assert len(images_np) == len(ref_np) == len(prompts_list), "images/ref_images/prompts 数量需一致"

        # 分批
        N = len(prompts_list)
        num_batches = math.ceil(N / batch_size)
        scores = []
        results = []

        for b in range(num_batches):
            s = b * batch_size
            e = min((b + 1) * batch_size, N)

            jpeg_images = []
            jpeg_ref_images = []
            batch_prompts = prompts_list[s:e]

            # 编码为 JPEG 字节
            for i in range(s, e):
                img = Image.fromarray(images_np[i])
                buf = BytesIO()
                img.save(buf, format="JPEG")
                jpeg_images.append(buf.getvalue())

                ref = Image.fromarray(ref_np[i])
                rbuf = BytesIO()
                ref.save(rbuf, format="JPEG")
                jpeg_ref_images.append(rbuf.getvalue())

            # 组织请求体（与服务端协议保持一致）
            data = {
                "output_images": jpeg_images,          # 当前编辑后的图片
                "images": jpeg_ref_images,  # 参考图片
                "prompts": batch_prompts,       # 对应的文本
            }
            payload = pickle.dumps(data)

            # 发请求
            resp = sess.post(url, data=payload, timeout=300)
            resp.raise_for_status()
            resp_obj = pickle.loads(resp.content)

            outputs = resp_obj.get("outputs", None)
            if outputs is None:
                # 兜底
                batch_scores = []
                batch_extra = {}
                batch_results = []
            else:
                batch_scores = list(outputs[0])
                batch_extra = outputs[1] if isinstance(outputs[1], dict) else {}
                batch_results = list(outputs[2]) if isinstance(outputs[2], (list, tuple)) else []
            # 兜底与清洗
            cleaned_scores = []
            for v in batch_scores:
                if isinstance(v, (int, float)) and np.isfinite(v):
                    cleaned_scores.append(float(v))
                else:
                    cleaned_scores.append(0.5)  # 默认分数
            scores.extend(cleaned_scores)
            results.extend(batch_results)

        # 返回与本地版保持相同签名：scores, extra_info, results
        return scores, {}, results

    return _fn

def masked_image_similarity_score(device="cuda", psnr_max_db: float = 40.0, psnr_weight: float = 0.3, ssim_weight: float = 0.35):
    """
    工厂函数，模仿你的 image_similarity_score 返回 _fn。
    使用:
      scorer_fn = masked_image_similarity_score(device)
      scores, extras = scorer_fn(images, ref_images, masks)
    """
    # scorer = MaskedSSIMCharbonnierScorer(device=device, psnr_weight=psnr_weight, ssim_weight=ssim_weight).to(device)
    scorer = MaskedPSNRSSIMScorer(device=device, psnr_max_db=psnr_max_db, ssim_weight=ssim_weight).to(device)


    def _fn(images, ref_images, masks):
        # 兼容非 torch.Tensor 的输入（与原函数一致）
        if not isinstance(images, torch.Tensor):
            images = np.asarray(images)
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        if not isinstance(ref_images, torch.Tensor):
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8) / 255.0
        # masks 也做相同处理
        if not isinstance(masks, torch.Tensor):
            masks = [np.array(m) for m in masks]
            masks = np.array(masks)
            if masks.ndim == 3:  # NHW -> NHWC 单通道
                masks = masks[..., None]
            masks = masks.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            masks = torch.tensor(masks, dtype=torch.uint8) / 255.0

        scores = scorer(images, ref_images, masks)
        return scores

    return _fn


def mllm_score_continue(device):
    """Submits images to GenEval and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = f"http://{os.getenv('REWARD_SERVER', 'localhost:12341')}/mode/logits_non_cot"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(ref_images, images, prompts, metadatas):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images_batched = np.array_split(images, np.ceil(len(images) / batch_size))

        if not isinstance(ref_images, torch.Tensor):
            ref_images = np.array([np.array(img) for img in ref_images])
            ref_images_batched = np.array_split(ref_images, np.ceil(len(ref_images) / batch_size))

        all_scores = []
        for image_batch, ref_image_batch in zip(images_batched, ref_images_batched):

            jpeg_images = []
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            ref_jpeg_images = []
            for ref_image in ref_image_batch:
                img = Image.fromarray(ref_image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                ref_jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "ref_images": ref_jpeg_images,
                "images": jpeg_images,
                "prompts": prompts,
                "metadatas": metadatas,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=360)
            response_data = pickle.loads(response.content)

            all_scores += response_data["scores"]

        return all_scores, {}

    return _fn

def aesthetic_score(device):
    from flow_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def clip_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def hpsv2_score(device):
    from flow_grpo.hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def imagereward_score(device):
    from flow_grpo.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def geneval_score(device):
    from flow_grpo.gen_eval import load_geneval

    batch_size = 64
    compute_geneval = load_geneval(device)

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            pil_images = [Image.fromarray(image) for image in image_batch]

            data = {
                "images": pil_images,
                "metadatas": list(metadata_batched),
                "only_strict": only_strict,
            }
            scores, rewards, strict_rewards, group_rewards, group_strict_rewards = compute_geneval(**data)

            all_scores += scores
            all_rewards += rewards
            all_strict_rewards += strict_rewards
            all_group_strict_rewards.append(group_strict_rewards)
            all_group_rewards.append(group_rewards)
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn


def ocr_score(device):
    from flow_grpo.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn


def unifiedreward_score_sglang(device):
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")

    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc / 5.0 for sc in score]
        return score, {}

    return _fn

def dummy():
    def _fn(images, prompts, metadata):
        return [random.random() for _ in range(len(images))], {}
    return _fn
    

def multi_score(device, score_dict):
    score_functions = {
        # "ocr": ocr_score,
        # "imagereward": imagereward_score,
        # "pickscore": pickscore_score,
        # "aesthetic": aesthetic_score,
        # "jpeg_compressibility": jpeg_compressibility,
        # "unifiedreward": unifiedreward_score_sglang,
        # "geneval": geneval_score,
        # "clipscore": clip_score,
        # "hpsv2": hpsv2_score,
        # "mllm_score_continue": mllm_score_continue,
        # "dummy": dummy,
        "mllm_score": mllm_score_remote,
        "pixel_consistency": masked_image_similarity_score,
    }
    score_fns = {}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = (
            score_functions[score_name](device)
            if "device" in score_functions[score_name].__code__.co_varnames
            else score_functions[score_name]()
        )

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, ref_images=None, only_strict=True, masks=None):
        total_scores = []
        score_details = {}

        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](
                    images, prompts, metadata, only_strict
                )
                score_details["accuracy"] = rewards
                score_details["strict_accuracy"] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f"{key}_strict_accuracy"] = value
                for key, value in group_rewards.items():
                    score_details[f"{key}_accuracy"] = value
            # elif score_name.startswith("mllm_"):
            #     scores, rewards = score_fns[score_name](ref_images, images, prompts, metadata)
            elif score_name == "mllm_score":
                scores, rewards, results = score_fns[score_name](images, ref_images, prompts)
            elif score_name == "pixel_consistency":
                scores = score_fns[score_name](images, ref_images, masks)
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details["avg"] = total_scores
        # {'mllm_score': [0.6, 0.6, 0.6], 
        # 'pixel_consistency': tensor([0.4494, 0.5014, 0.4937], device='cuda:6'), 
        # 'avg': [tensor(0.5548, device='cuda:6'), tensor(0.5704, device='cuda:6'), tensor(0.5681, device='cuda:6')]}
        # print(score_details)
        return score_details, {}

    return _fn


# def main():
#     import torchvision.transforms as transforms

#     image_paths = [
#         "test_cases/nasa.jpg",
#     ]

#     transform = transforms.Compose(
#         [
#             transforms.ToTensor(),  # Convert to tensor
#         ]
#     )

#     images = torch.stack([transform(Image.open(image_path).convert("RGB")) for image_path in image_paths])
#     prompts = [
#         'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
#     ]
#     metadata = {}  # Example metadata
#     score_dict = {"unifiedreward": 1.0}
#     # Initialize the multi_score function with a device and score_dict
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     scoring_fn = multi_score(device, score_dict)
#     # Get the scores
#     scores, _ = scoring_fn(images, prompts, metadata)
#     # Print the scores
#     print("Scores:", scores)

def main():
    import torchvision.transforms as transforms

    image_paths = [
        "/home/notebook/data/group/wyh/Step1X-Edit/examples/0000.jpg",
        "/home/notebook/data/group/wyh/Step1X-Edit/examples/0001.png",
        "/home/notebook/data/group/wyh/Step1X-Edit/examples/0002.jpg",
        "/home/notebook/data/group/wyh/Step1X-Edit/examples/0003.png",
        "/home/notebook/data/group/wyh/Step1X-Edit/examples/0004.jpg",
    ]
    ref_image_paths = [
        "/home/notebook/data/group/wyh/Step1X-Edit/results/output_cn-1024/0000.jpg",
        "/home/notebook/data/group/wyh/Step1X-Edit/results/output_cn-1024/0001.png",
        "/home/notebook/data/group/wyh/Step1X-Edit/results/output_cn-1024/0002.jpg",
        "/home/notebook/data/group/wyh/Step1X-Edit/results/output_cn-1024/0003.png",
        "/home/notebook/data/group/wyh/Step1X-Edit/results/output_cn-1024/0004.jpg",
    ]

    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # Convert to tensor
    # ])

    # images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    # ref_images = torch.stack([transform(Image.open(ref_image_path).convert('RGB')) for ref_image_path in ref_image_paths])
    # 先把较短边缩放到≥512，再中心裁成 512x512
    preprocess = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32)  # 可选：若后续期望 float32
    ])

    # transform = transforms.Compose([
    #     transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),  # 等比把短边缩放到 512
    #     transforms.CenterCrop(512),  # 中心裁成 512x512
    #     transforms.ToTensor(),       # [0,1] 的张量，形状 CxHxW
    # ])
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),  # 直接缩放到 512x512
        transforms.ToTensor(),  # [0,1] 的张量，形状 CxHxW
    ])

    def load_and_preprocess(path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return transform(img)

    images = torch.stack([load_and_preprocess(p) for p in image_paths])        # [N,3,512,512]
    ref_images = torch.stack([load_and_preprocess(p) for p in ref_image_paths])  # [N,3,512,512]

    prompts=[
        "Add pendant with a ruby around this girl's neck.",
        "Let her cry.",
        "Change the outerwear to be made of top-grain calfskin.",
        "Change image to anime style.",
        "Replace 'TRAIN' with 'PLANE'"
    ]
    metadata = {}  # Example metadata
    score_dict = {
        # "unifiedreward": 1.0
        "mllm_score": 1.0
    }
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    import time
    t1 = time.time()
    scores, rewards = scoring_fn(images, prompts, metadata, ref_images)
    print(time.time() - t1)
    # scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)
    # print(results)



if __name__ == "__main__":
    main()
