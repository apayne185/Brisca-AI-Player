# syntax=docker/dockerfile:1.7

# --- Build: resolve the locked environment with uv -------------------------
# The policy is served through ONNX Runtime, so PyTorch is not installed.
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first, so code changes don't invalidate this layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --extra onnx --extra detect --extra serve

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra onnx --extra detect --extra serve

# --- Runtime: slim image, non-root user ------------------------------------
FROM python:3.12-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 brisca
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY models ./models
ENV PATH="/app/.venv/bin:$PATH" \
    BRISCA_MODELS_DIR=/app/models \
    PYTHONUNBUFFERED=1
USER brisca
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"
CMD ["uvicorn", "brisca.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
