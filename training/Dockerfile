# Build from repository root: docker build -f training/Dockerfile -t v2d-policy .
# Run with a committed clone mounted at /workspace; GPU host driver >=580.
FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv git git-lfs ca-certificates libgl1 libegl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/bootstrap-env && /opt/bootstrap-env/bin/pip install --no-cache-dir uv
ENV PATH="/opt/bootstrap-env/bin:${PATH}"
WORKDIR /workspace
COPY training /workspace/training
RUN python -m training.scripts.bootstrap --runtime /opt/v2d-runtime --download-assets \
    && python -m training.scripts.setup_runtime --runtime /opt/v2d-runtime \
        --venv /opt/v2d-policy-env --profile cuda13 --driver-major 580 > /tmp/native-setup.sh \
    && bash -e /tmp/native-setup.sh
ENV PATH="/opt/v2d-policy-env/bin:${PATH}" XLA_PYTHON_CLIENT_PREALLOCATE=false
ENTRYPOINT ["bash", "/workspace/training/scripts/container_entrypoint.sh"]
CMD ["python", "-m", "training.scripts.pipeline", "--help"]
