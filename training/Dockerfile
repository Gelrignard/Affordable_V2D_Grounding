# Build with: docker build -f training/Dockerfile -t v2d-policy:pilot training
# Requires a host NVIDIA driver compatible with the pinned runtime (>=580).
FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv git git-lfs ca-certificates libgl1 libegl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" XLA_PYTHON_CLIENT_PREALLOCATE=false
WORKDIR /workspace
COPY . /workspace/training
RUN python -m training.bootstrap --runtime /opt/v2d-runtime --download-assets \
    && pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir /opt/v2d-runtime/robotic_grounding/flash_chord
ENV PYTHONPATH="/workspace:/opt/v2d-runtime/robotic_grounding/flash_chord/src"
CMD ["python", "-m", "training", "doctor", "--runtime", "/opt/v2d-runtime"]
