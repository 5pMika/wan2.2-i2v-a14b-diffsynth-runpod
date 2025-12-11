import base64
import io
import logging
import os
import tempfile
import uuid
from typing import Optional, Tuple

import requests
import runpod
import torch
from PIL import Image
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wan2.2-i2v")

MODEL_ID = os.getenv("WAN_MODEL_ID", "PAI/Wan2.2-I2V-A14B")
TOKENIZER_ID = os.getenv("WAN_TOKENIZER_ID", "Wan-AI/Wan2.1-T2V-1.3B")

PIPELINE: Optional[WanVideoPipeline] = None


def _vram_config() -> dict:
    dtype = torch.bfloat16
    return {
        "offload_dtype": dtype,
        "offload_device": "cpu",
        "onload_dtype": dtype,
        "onload_device": "cpu",
        "preparing_dtype": dtype,
        "preparing_device": "cuda" if torch.cuda.is_available() else "cpu",
        "computation_dtype": dtype,
        "computation_device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def _get_vram_limit() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        # leave a small safety buffer
        free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
        return max(free_gb - 2, 1.0)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not read CUDA memory info: %s", exc)
        return None


def _load_pipeline() -> WanVideoPipeline:
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vram_cfg = _vram_config()

    model_configs = [
        ModelConfig(
            model_id=MODEL_ID,
            origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
            **vram_cfg,
        ),
        ModelConfig(
            model_id=MODEL_ID,
            origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
            **vram_cfg,
        ),
        ModelConfig(
            model_id=MODEL_ID,
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            **vram_cfg,
        ),
        ModelConfig(
            model_id=MODEL_ID,
            origin_file_pattern="Wan2.1_VAE.pth",
            **vram_cfg,
        ),
    ]

    tokenizer_config = ModelConfig(
        model_id=TOKENIZER_ID,
        origin_file_pattern="google/umt5-xxl/",
        **vram_cfg,
    )

    logger.info("Loading WanVideoPipeline on %s", device)
    PIPELINE = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        vram_limit=_get_vram_limit(),
    )
    return PIPELINE


def _decode_base64_image(b64_str: str) -> Image.Image:
    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[-1]
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data))


def _load_image(image_ref: str, size: Tuple[int, int]) -> Image.Image:
    if image_ref.startswith("http://") or image_ref.startswith("https://"):
        resp = requests.get(image_ref, timeout=30)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
    elif os.path.exists(image_ref):
        img = Image.open(image_ref)
    else:
        img = _decode_base64_image(image_ref)
    return img.convert("RGB").resize(size)


def _download_to_temp(url: str, suffix: str) -> str:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(tmp_fd, "wb") as f:
        f.write(resp.content)
    return tmp_path


def _maybe_upload_s3(local_path: str, job_id: str) -> Optional[str]:
    bucket = os.getenv("BUCKET_NAME")
    endpoint = os.getenv("ENDPOINT_URL")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not all([bucket, endpoint, access_key, secret_key]):
        return None
    try:
        import boto3

        session = boto3.session.Session()
        s3 = session.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        key = f"wan2.2-i2v/{job_id}.mp4"
        s3.upload_file(local_path, bucket, key)
        return f"{endpoint}/{bucket}/{key}"
    except Exception as exc:  # pragma: no cover
        logger.warning("Upload to S3 failed: %s", exc)
        return None


def handler(job):
    job_input = job.get("input", {})
    job_id = str(job.get("id", uuid.uuid4()))

    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "prompt is required"}

    image_ref = job_input.get("image")
    if not image_ref:
        return {"error": "image is required (url, base64 or path)"}

    negative_prompt = job_input.get("negative_prompt", "")
    height = int(job_input.get("height", 480))
    width = int(job_input.get("width", 832))
    tiled = bool(job_input.get("tiled", True))
    fps = int(job_input.get("fps", 15))
    quality = int(job_input.get("quality", 5))
    seed = int(job_input.get("seed", 1))

    control_video_path = None
    if job_input.get("control_video"):
        control_video_path = (
            _download_to_temp(job_input["control_video"], suffix=".mp4")
            if job_input["control_video"].startswith("http")
            else job_input["control_video"]
        )

    ref_image = _load_image(image_ref, (width, height))
    control_video = (
        VideoData(control_video_path, height=height, width=width)
        if control_video_path
        else None
    )

    pipe = _load_pipeline()

    with torch.inference_mode():
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            vace_reference_image=ref_image,
            vace_video=control_video,
            seed=seed,
            tiled=tiled,
        )

    output_path = os.path.join(tempfile.gettempdir(), f"{job_id}.mp4")
    save_video(video, output_path, fps=fps, quality=quality)

    s3_url = _maybe_upload_s3(output_path, job_id)

    return {
        "output_path": output_path,
        "s3_url": s3_url,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "fps": fps,
        "tiled": tiled,
    }


runpod.serverless.start({"handler": handler})
