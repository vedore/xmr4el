FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    procps \
    build-essential \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USER=xmr4el \
    HOME=/tmp

# Install dependencies outside the source mount; code is supplied at runtime.
WORKDIR /opt/deps
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project --no-cache

# Use PYTHONPATH, not the script directory (test/xmr4el shadows the package).
ENV PYTHONSAFEPATH=1
WORKDIR /app
CMD ["/bin/bash"]
