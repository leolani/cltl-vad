# syntax = docker/dockerfile:1.2

FROM python:3.9

WORKDIR /cltl-vad
COPY src requirements.txt makefile ./
COPY config ./config
COPY util ./util

RUN --mount=type=bind,target=/cltl-vad/repo,from=cltl/cltl-requirements:latest,source=/repo \
        make venv project_repo=/cltl-vad/repo/leolani project_mirror=/cltl-vad/repo/mirror

CMD . venv/bin/activate && python app.py