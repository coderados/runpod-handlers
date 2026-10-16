FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN pip install --no-cache-dir \
    "diffusers>=0.34.0" \
    transformers accelerate safetensors sentencepiece \
    "imageio[ffmpeg]" pillow requests runpod

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
