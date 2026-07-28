# Build from the repo root: docker buildx build -f deploy/Dockerfile .
FROM python:3.14-slim

WORKDIR /app

COPY pyproject.toml /app/
COPY hyh/ /app/hyh/
COPY actions/ /app/actions/
COPY runner/ /app/runner/

RUN pip install --no-cache-dir ".[claude]" psutil

COPY deploy/config.yaml /app/config.yaml

# Containers run with stdin closed, so the terminal channel closes at startup
# and the harness serves telegram. That is the intended deployment mode.
CMD ["python", "-m", "hyh.main", "--chat"]
