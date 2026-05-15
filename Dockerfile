# syntax = docker/dockerfile:1.2

FROM python:3.9

WORKDIR /cltl-vad
COPY src requirements.txt makefile ./
COPY config ./config
COPY util ./util

RUN --mount=type=bind,target=/cltl-vad/repo,from=cltl/cltl-requirements:latest,source=/repo \
        make venv project_repo=/cltl-vad/repo/leolani project_mirror=/cltl-vad/repo/mirror

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD . venv/bin/activate && python main.py