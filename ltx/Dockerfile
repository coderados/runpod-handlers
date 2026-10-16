FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "diffusers>=0.37.0" \
    "transformers>=4.45.0" accelerate safetensors sentencepiece \
    "imageio[ffmpeg]" pillow requests runpod soundfile

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
