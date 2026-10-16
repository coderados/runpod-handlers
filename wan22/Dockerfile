FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

RUN pip install --no-cache-dir \
    "diffusers>=0.34.0" \
    "transformers>=4.51.0,<5" accelerate safetensors sentencepiece \
    "imageio[ffmpeg]" pillow requests runpod ftfy

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
