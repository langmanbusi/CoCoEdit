import os
import math
import gc
import pickle
import traceback
from io import BytesIO
from typing import List, Tuple

import numpy as np
from PIL import Image
from flask import Flask, request, Blueprint, Response

import ray

# ===================== 配置 =====================
# GPU 总数与每个 Actor 占用的 GPU 数（若模型需要张量并行可设 >1）
NUM_GPUS = int(os.getenv("NUM_GPUS", "8"))
NUM_TP = int(os.getenv("NUM_TP", "1"))  # 每个 Actor 占用的 GPU 数
MAX_ACTORS = max(1, NUM_GPUS // max(1, NUM_TP))

# 单次请求的最大批大小（可用客户端也分批，二者叠加）
SERVER_BATCH = int(os.getenv("SERVER_BATCH", "16"))

# Flask/Gunicorn 侧最大请求体（如经由 nginx，还需 nginx 同步放大）
MAX_CONTENT_LENGTH_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "64"))

# ===================== Flask 应用 =====================
root = Blueprint("root", __name__)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024

# ===================== 业务推理函数封装为 Ray Actor =====================
# 你原来的加载函数
# from reward_server.editscore_eval import load_editscore_0_5_EditScorelora as load_editscore
from reward_server.editscore_eval import load_editscore_0_5 as load_editscore

def _decode_images(jpeg_bytes_list: List[bytes]) -> List[Image.Image]:
    images = []
    for d in jpeg_bytes_list:
        im = Image.open(BytesIO(d))
        im.load()  # 读取到内存
        images.append(im.convert("RGB"))
    return images

@ray.remote(num_gpus=NUM_TP)
class EditScoreWorker:
    def __init__(self):
        # 在 Actor 内加载模型，绑定到 Ray 分配的 GPU
        self.infer = load_editscore()

    def evaluate(self, images_jpeg: List[bytes], outputs_jpeg: List[bytes], prompts: List[str]):
        # 每个 Actor 内部可再做更细分批，避免峰值 OOM
        imgs = _decode_images(images_jpeg)
        outs = _decode_images(outputs_jpeg)
        # 直接调用你原推理函数
        result = self.infer(imgs, outs, prompts)
        # 返回可 pickle 的对象
        return result

# ============== Ray 初始化与 Actor 池 ==============
WORKERS = None  # List[ActorHandle]

def init_ray_and_workers():
    global WORKERS
    if not ray.is_initialized():
        # address="auto" 可接入已有集群；默认本机
        ray.init(ignore_reinit_error=True)
    if WORKERS is None:
        WORKERS = [EditScoreWorker.remote() for _ in range(MAX_ACTORS)]
        print(f"Initialized {len(WORKERS)} Ray workers (NUM_GPUS={NUM_GPUS}, NUM_TP={NUM_TP})")
    return WORKERS

# ============== 简单的轮转调度器 ==============
class RoundRobin:
    def __init__(self, n: int):
        self.n = n
        self.i = 0
    def next(self) -> int:
        j = self.i
        self.i = (self.i + 1) % self.n
        return j

RR = None

# ===================== 健康检查 =====================
@root.route("/health", methods=["GET"])
def health():
    ok = (WORKERS is not None)
    return Response("ok" if ok else "loading", status=200 if ok else 503)

# ===================== 主推理路由 =====================
@root.route("/", methods=["POST"])
def inference():
    global RR
    try:
        payload = pickle.loads(request.get_data())
        images_bytes = payload["images"]
        output_images_bytes = payload["output_images"]
        prompts = payload["prompts"]

        if not isinstance(images_bytes, list) or not isinstance(output_images_bytes, list) or not isinstance(prompts, list):
            raise ValueError("images/output_images/prompts must be lists")
        if not (len(images_bytes) == len(output_images_bytes) == len(prompts)):
            raise ValueError("Length mismatch among images/output_images/prompts")

        N = len(prompts)
        # 按 SERVER_BATCH 在服务端再切一次批，避免某些客户端批过大
        num_batches = math.ceil(N / SERVER_BATCH)

        if WORKERS is None:
            init_ray_and_workers()
        if RR is None:
            RR = RoundRobin(len(WORKERS))

        obj_refs = []
        batch_slices: List[Tuple[int, int]] = []

        for b in range(num_batches):
            s = b * SERVER_BATCH
            e = min((b + 1) * SERVER_BATCH, N)
            worker = WORKERS[RR.next()]
            # 提交到不同 Actor 异步执行
            ref = worker.evaluate.remote(images_bytes[s:e], output_images_bytes[s:e], prompts[s:e])
            obj_refs.append(ref)
            batch_slices.append((s, e))

        # 收集结果并拼接（假设你的 infer 返回结构为 (scores, extra, results) 或类似）
        results = ray.get(obj_refs)

        # 聚合逻辑：根据你的返回结构定义
        # 这里做一个“尽量兼容”的合并：outputs[0]=scores(list), outputs[1]=dict(合并), outputs[2]=results(list)
        merged_scores = []
        merged_extra = {}
        merged_results = []
        for out in results:
            if isinstance(out, (list, tuple)) and len(out) >= 1:
                # scores
                scores = list(out[0]) if out[0] is not None else []
                merged_scores.extend(scores)
                # extra
                if len(out) >= 2 and isinstance(out[1], dict):
                    # 简单合并（后者覆盖前者）
                    merged_extra.update(out[1])
                # results
                if len(out) >= 3 and isinstance(out[2], (list, tuple)):
                    merged_results.extend(list(out[2]))
            else:
                # 如果是单列表也收下
                if isinstance(out, list):
                    merged_scores.extend(out)

        resp_obj = {"outputs": (merged_scores, merged_extra, merged_results)}
        resp_bytes = pickle.dumps(resp_obj)

        # 降低内存峰值
        del images_bytes, output_images_bytes, prompts, payload, results
        gc.collect()

        return Response(resp_bytes, status=200, mimetype="application/octet-stream")

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return Response(tb.encode("utf-8"), status=500, mimetype="text/plain")

def create_app():
    # Flask 工厂：让 gunicorn worker 进程内创建 Flask。Ray/模型延迟到首次请求时初始化。
    application = Flask(__name__)
    application.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024
    application.register_blueprint(root)
    return application

# 直接 python 运行
if __name__ == "__main__":
    init_ray_and_workers()
    print(f"Starting Flask server with {len(WORKERS)} Ray workers...")
    app.register_blueprint(root)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "18086")), debug=False)