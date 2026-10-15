FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/eqgpt
COPY . .
ENV PYTHONPATH=/workspace/eqgpt/src TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLBACKEND=Agg
RUN python -m pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121 && python -m pip install --no-cache-dir -r docs/eqgpt/local_environment.lock.txt && python -m pip install --no-deps -e .
CMD ["python", "scripts/eqgpt/preflight.py", "--require-cuda"]
