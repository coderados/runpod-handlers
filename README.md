# RunPod Serverless Handlers — Wan 2.2 & LTX 2.3

Two serverless worker images for image-to-video, sized for the 48GB tier
(A6000 Ada / A40 / L40, $1.22/hr ≈ $0.00034/s, billed only while working).

## Models

| Dir    | Model | Notes |
|--------|-------|-------|
| `wan22/` | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 14B MoE, best quality, ~30-40GB VRAM with CPU offload. 16 fps. |
| `ltx/`   | `FastVideo/LTX-2.3-Distilled-Diffusers` | Distilled fast variant, 8 steps, generates audio too. 24 fps. |

Override with env `MODEL_ID` (e.g. `Lightricks/LTX-2.3` for the full model).
Env `STEPS` overrides inference steps (Wan default 30, LTX default 8).

## Request format (RunPod /run input)

```json
{
  "input": {
    "image": "data:image/png;base64,... or https://... (optional)",
    "prompt": "camera slowly pushes in...",
    "duration": 5,
    "aspect_ratio": "16:9",
    "seed": 42,
    "save_audio": true
  }
}
```

Returns `{ "video_b64": "<base64 mp4>", "meta": {...} }` on success,
`{ "error": "..." }` on failure.

## Build & push

You need a Docker registry (Docker Hub works free):

```bash
docker build -t YOURUSER/wan22-i2v:latest runpod-handlers/wan22
docker push YOURUSER/wan22-i2v:latest

docker build -t YOURUSER/ltx23-fast-i2v:latest runpod-handlers/ltx
docker push YOURUSER/ltx23-fast-i2v:latest
```

## Create the endpoints

In the RunPod console → Serverless → New Endpoint:

- **GPU**: 48GB tier (A6000 Ada recommended, A40 cheapest)
- **Container image**: the image you pushed
- **Container disk**: 80GB+ (model weights download on first cold start)
- **Workers**: min 0 (scale-to-zero = pay only when generating), max 1-2
- **FlashBoot**: enable if offered (faster cold starts)
- Set `HF_TOKEN` env if the model repo ever requires auth

Then copy the endpoint ID into Pruna Video Studio:
Backend → "LTX (self-hosted)" or "Wan 2.2 (self-hosted)" → paste either
`https://api.runpod.ai/v2/<ENDPOINT_ID>` or just the endpoint ID.
The app uses the server-side `RUNPOD_API_KEY` from `/workspace/.env`.

## Cost notes

- Billed per second only while a worker runs: cold start + generation + idle timeout.
- Attach a network volume mounted at `/runpod-volume` and the HF cache persists
  across cold starts (weights download once, not per worker).
- Typical: LTX 5s clip ≈ 20-40s GPU time (~$0.01), Wan 5s 720p ≈ 2-5 min (~$0.05-0.10).
