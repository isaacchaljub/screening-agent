# syntax=docker/dockerfile:1
#
# Two stages so the vendor SDKs' build-time toolchain (chromadb pulls native wheels) never
# reaches the runtime image, and so a code change re-uses the cached dependency layer instead of
# reinstalling ~400MB of wheels. `pyproject.toml` is copied on its own, before `src/`, for exactly
# that reason — the dependency layer is keyed on it alone.

# --------------------------------------------------------------------------------------------
FROM python:3.11-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies first: this layer is invalidated only by a pyproject change, not by editing code.
#
# torch is installed from the CPU-only index BEFORE the project, so pip's resolver already has it
# satisfied and never pulls the default wheel — which bundles CUDA and is several GB. Inference
# here is a 118M-parameter embedding model on 40 short chunks; there is no GPU to use even if the
# image carried the drivers for one.
COPY pyproject.toml ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
RUN mkdir -p src/screening_agent \
    && touch src/screening_agent/__init__.py README.md \
    && pip install .

COPY src/ ./src/
RUN pip install --no-deps .

# Bake the embedding model into the image rather than downloading it on first use. Three reasons:
# a cold container would otherwise pay a ~470MB download before answering its first FAQ question;
# the runtime would need outbound access to huggingface.co, which a locked-down deployment may not
# have; and a Hugging Face outage would become a runtime failure in a system whose whole point is
# that embeddings no longer depend on somebody else's uptime.
ARG EMBED_MODEL=intfloat/multilingual-e5-small
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EMBED_MODEL')"

# --------------------------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Never run the app as root. The volume mount point is created and chowned here rather than left
# to Docker, which would otherwise create it root-owned on first `-v` mount and make the SQLite
# write fail at the first turn.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data \
    && chown -R app:app /app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=demo

COPY --from=build /opt/venv /opt/venv
# The model cache the build stage just populated. HF_HOME points the runtime at it, and
# HF_HUB_OFFLINE makes a missing model fail loudly at startup instead of silently reaching out to
# the network — the failure mode this whole change exists to remove.
COPY --from=build --chown=app:app /root/.cache/huggingface /home/app/.cache/huggingface
ENV HF_HOME=/home/app/.cache/huggingface \
    HF_HUB_OFFLINE=1

WORKDIR /app
USER app

# `data/` holds the SQLite database, the per-conversation JSON exports, and the Chroma index.
# All three are runtime state, deliberately not baked into the image — mount a volume to keep
# them across `docker run`s: `-v "$PWD/data:/app/data"`.
VOLUME ["/app/data"]

EXPOSE 8000

# Matches api.py's own liveness endpoint. Uses urllib rather than curl so the runtime image
# doesn't need to carry curl just to answer a health check.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2).status == 200 else 1)"

# One worker on purpose. `api.py` keeps in-flight `Conversation` objects in a module-level dict
# (turn-scoped state — attempts, history — that isn't reconstructible from SQLite alone), so a
# second worker would serve some requests from a process that has never seen the conversation.
# Moving that state to Redis is the first step to scaling out — see docs/deployment.md.
CMD ["uvicorn", "screening_agent.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
