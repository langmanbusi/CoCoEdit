import imp
import os
import datetime


base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    run_name = os.getenv("RUN_NAME", "")
    return globals()[name](run_name=run_name)

def _get_config(base_model="qwen_image_edit", n_gpus=1, gradient_step_per_epoch=1, reward_fn={}, name="", num_image=12, bsz=3, num_groups=24):
    config = base.get_config()

    # config.wandb_name = "exp_editscore_psnrssim_cps_0"

    config.base_model = base_model
    config.transformer_path = None
    config.dataset = "dataset/editing" # the folder include jsonl file
    config.train_json_name = "metadata.jsonl"
    config.test_json_name = "metadata.jsonl"
    
    config.pretrained.model = "ckpt/Qwen-Image-Edit-2509"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 10
    config.sample.guidance_scale = 4.0
    config.log_image_freq = 10
    config.resolution = 512
    config.train.beta = 0.0001
    config.sample.noise_level = 0.7
    bsz = bsz  # 3

    config.sample.num_image_per_prompt = num_image  # 每个样本产生多少个输出,groupsize

    config.sample.ban_std_thres = 0.05
    config.sample.ban_mean_thres = 0.9
    config.sample.ban_prompt = False
    num_groups = num_groups  # 一次epoch中不同的样本数量
    
    # 确保一次epoch中，不同的样本产生的总输出个数能够被gpu*bsz整除，并且gpu*bsz能被输出个数整除（商为机器上同时处理的样本数量）
    while True:
        if bsz < 1:
            assert False, "Cannot find a proper batch size."
        if (
            num_groups * config.sample.num_image_per_prompt % (n_gpus * bsz) == 0
            and bsz * n_gpus % config.sample.num_image_per_prompt == 0
        ):
            n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
            if n_batch_per_epoch % gradient_step_per_epoch == 0:
                config.sample.train_batch_size = bsz
                config.sample.num_batches_per_epoch = n_batch_per_epoch
                config.train.batch_size = config.sample.train_batch_size
                config.train.gradient_accumulation_steps = (
                    config.sample.num_batches_per_epoch // gradient_step_per_epoch
                )
                break
        bsz -= 1

    # special design, the test set has a total of 1018/2212/2048 for ocr/geneval/pickscore, to make gpu_num*bs*n as close as possible to it, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.test_batch_size = bsz
    if n_gpus > 32:
        config.sample.test_batch_size = config.sample.test_batch_size // 2

    config.prompt_fn = "geneval"

    config.run_name = f"nft_{base_model}_{name}"
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.save_dir = f"ckpts/nft_{base_model}_{name}_{unique_id}"
    config.reward_fn = reward_fn

    config.decay_type = 1
    config.beta = 1.0
    config.train.adv_mode = "all"

    # config.sample.guidance_scale = 1.0
    config.sample.deterministic = True
    config.sample.solver = "dpm2"

    config.use_reg = False
    config.use_reg_xsrc = False
    return config

def qwenedit_qwen32_pixel_reward_reg(run_name='exp_qwenedit'):
    run_name = "exp_nft_qwen32_pixel_ratio7-3_cfg1_newver_reg_ratio8-2"
    reward_fn = {
        # "mllm_score_continue": 1.0,
        "pixel_consistency": 0.2,
        "mllm_score": 0.8,
    }
    config = _get_config(
        base_model="qwen_image_edit",
        n_gpus=16,
        gradient_step_per_epoch=1,
        reward_fn=reward_fn,
        name=run_name,
        num_image=12,
        bsz=3,
        num_groups=24,
    )
    # number of epochs between saving model checkpoints.
    config.save_freq = 30
    config.eval_freq = 30

    config.sample.ban_prompt = True
    config.sample.ban_std_thres = 0.02
    config.debug = False
    config.sample.guidance_scale = 1.0
    
    config.use_reg = True
    config.train.lambda_bg_pos = 0.8
    config.train.lambda_ed_neg = 0.5
    # config.train.lambda_bg_neg = 1.0
    # config.train.bg_weight_neg = 0.2
    config.use_reg_xsrc = True
    config.train.ema = False

    return config

