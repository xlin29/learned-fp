# LearnedFP evaluation pipeline — Python + PyTorch CPU.
#
# This image is sized for the demo and code review. To run paper-scale
# training (paper §5.2 / §5.4), override torch with the matching CUDA
# wheel and provision a GPU (the paper used 1× A100).
#
# Build:
#   cd pipeline
#   docker build -t learnedfp-pipeline .
#
# Demo (5 min, CPU) — override the cli.py entrypoint with the smoke-test
# script (use `bash` so the host file's exec bit doesn't matter — zip
# distributions of the artifact strip +x. Must be bash, not sh, because
# demo.sh uses set -o pipefail which the debian-slim /bin/sh rejects):
#   docker run --rm -v "$(pwd)/..:/workspace" \
#     --entrypoint bash learnedfp-pipeline /workspace/pipeline/demo.sh

FROM python:3.11-slim-bookworm

# Compiler + headers for any source-built deps. ca-certificates is
# needed because some pip wheels validate certs during install.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential pkg-config ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /pipeline

# Install Python deps. Use the CPU wheel index for torch so we don't
# pull a multi-GB CUDA payload by accident on a CPU build host.
COPY requirements.txt .
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

# Pipeline source.
COPY . /pipeline/
RUN chmod +x /pipeline/demo.sh /pipeline/cli.py

# CLI is the canonical entry point.
ENTRYPOINT ["python", "/pipeline/cli.py"]
CMD ["--help"]
