import base64
import io
import os
import subprocess
import tempfile

import requests
import runpod
import torch
from diffusers import LTX2ImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image

MODEL_ID = os.environ.get("MODEL_ID", "FastVideo/LTX-2.3-Distilled-Diffusers")
DEFAULT_FPS = 24

pipe = None


def init():
    global pipe
    pipe = LTX2ImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
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
    "16:9": (1216, 704),
    "9:16": (704, 1216),
    "4:3": (928, 704),
    "3:4": (704, 928),
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
    num_frames = max(9, (num_frames // 8) * 8 + 1)

    kwargs = dict(
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        frame_rate=float(DEFAULT_FPS),
        num_inference_steps=int(os.environ.get("STEPS", "8")),
        generator=generator,
        output_type="np",
    )
    if image is not None:
        kwargs["image"] = image

    result = pipe(**kwargs)
    frames = result.frames[0]
    audio = getattr(result, "audio", None)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    export_to_video(frames, path, fps=DEFAULT_FPS)

    if audio is not None and inp.get("save_audio", True):
        muxed = mux_audio(path, audio)
        if muxed:
            path = muxed

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


def mux_audio(video_path, audio):
    try:
        import numpy as np
        import soundfile as sf

        a = audio
        if torch.is_tensor(a):
            a = a.detach().cpu().float().numpy()
        a = np.asarray(a)
        if a.ndim == 3:
            a = a[0]
        if a.ndim == 2 and a.shape[0] in (1, 2):
            a = a.T
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as af:
            wav_path = af.name
        sf.write(wav_path, a, 24000)
        out_path = video_path.replace(".mp4", "-av.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-i", wav_path, "-c:v", "copy", "-c:a", "aac", "-shortest", out_path],
            check=True, capture_output=True,
        )
        os.unlink(video_path)
        os.unlink(wav_path)
        return out_path
    except Exception:
        return None


init()
runpod.serverless.start({"handler": handler})
