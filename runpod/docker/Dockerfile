# syntax=docker/dockerfile:1.7

ARG UV_IMAGE="ghcr.io/astral-sh/uv@sha256:4d881648806e80096429b24963908ecf5cb249605bb6800a3fe100a93c49ea3a"
ARG BASE_IMAGE="docker.io/nvidia/cuda@sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9"

FROM ${UV_IMAGE} AS uv
FROM ${BASE_IMAGE}

ARG VOXDELTA_GIT_COMMIT
ARG VOXDELTA_RUNPOD_LOCK_SHA256
ARG VOXDELTA_CODE_SHA256

LABEL org.opencontainers.image.title="VoxDelta RunPod XLS-R trainer" \
      org.opencontainers.image.revision="${VOXDELTA_GIT_COMMIT}" \
      io.voxdelta.runpod.lock-sha256="${VOXDELTA_RUNPOD_LOCK_SHA256}" \
      io.voxdelta.runpod.platform="linux/amd64"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/voxdelta/runpod/.venv/bin:${PATH}" \
    PYTHONPATH="/opt/voxdelta/backend/src" \
    VOXDELTA_CODE_SHA256="${VOXDELTA_CODE_SHA256}" \
    VOXDELTA_GIT_COMMIT="${VOXDELTA_GIT_COMMIT}"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libsndfile1 \
        python3.12 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /opt/voxdelta
COPY backend/pyproject.toml backend/uv.lock ./backend/
COPY runpod/pyproject.toml runpod/uv.lock ./runpod/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --project runpod --frozen --no-dev --no-editable --no-install-project

COPY backend/src ./backend/src
COPY runpod/src ./runpod/src
COPY runpod/config ./runpod/config
COPY runpod/scripts ./runpod/scripts

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --project runpod --frozen --no-dev --no-editable \
    && find /opt/voxdelta -type d -name __pycache__ -prune -exec rm -rf '{}' +

WORKDIR /workspace/voxdelta
ENTRYPOINT ["/opt/voxdelta/runpod/.venv/bin/python"]
CMD ["-c", "from voxdelta_runpod.config import CANONICAL_LABELS; print('voxdelta-runpod-ready', len(CANONICAL_LABELS))"]
