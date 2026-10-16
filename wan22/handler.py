import base64
import io
import os
import subprocess
import tempfile

import requests
import runpod
import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image

MODEL_ID = os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
DEFAULT_FPS = 16

pipe = None


def init():
    global pipe
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()


def load_image(ref):
    if not ref:
        return None
    if ref.startswith("data:image/"):
        b64 = ref.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    resp = requests.get(ref, timeout=60)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


SIZES = {
    "16:9": (1280, 704),
    "9:16": (704, 1280),
    "4:3": (960, 720),
    "3:4": (720, 960),
    "3:2": (1056, 704),
    "2:3": (704, 1056),
    "1:1": (960, 960),
}


def handler(job):
    inp = job.get("input") or {}
    prompt = (inp.get("prompt") or "").strip()
    if not prompt:
        return {"error": "prompt is required"}

    image = load_image(inp.get("image"))
    duration = int(inp.get("duration") or 5)
    duration = max(1, min(20, duration))
    aspect = inp.get("aspect_ratio") or "16:9"
    width, height = SIZES.get(aspect, SIZES["16:9"])
    if image is not None:
        width, height = image.size
        width -= width % 32
        height -= height % 32
    seed = inp.get("seed")
    generator = torch.Generator(device="cpu").manual_seed(int(seed)) if seed not in (None, "") else None

    num_frames = duration * DEFAULT_FPS
    num_frames = max(9, (num_frames // 4) * 4 + 1)

    kwargs = dict(
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=int(os.environ.get("STEPS", "30")),
        guidance_scale=3.5,
        generator=generator,
        output_type="np",
    )
    if image is not None:
        kwargs["image"] = image

    result = pipe(**kwargs)
    frames = result.frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    export_to_video(frames, path, fps=DEFAULT_FPS)

    with open(path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(path)

    return {
        "video_b64": video_b64,
        "meta": {
            "model": MODEL_ID,
            "frames": num_frames,
            "fps": DEFAULT_FPS,
            "width": width,
            "height": height,
        },
    }


init()
runpod.serverless.start({"handler": handler})
