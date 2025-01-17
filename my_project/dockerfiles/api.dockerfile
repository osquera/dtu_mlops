# Base image
FROM python:3.12-slim

EXPOSE $PORT

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /


COPY requirements_api.txt requirements_api.txt
COPY pyproject_api.toml pyproject_api.toml
COPY src/my_project/api.py api.py
COPY models/model.onnx /models/model.onnx

RUN mv pyproject_api.toml pyproject.toml

RUN pip install -r requirements_api.txt --no-cache-dir
RUN pip install pydantic

CMD exec uvicorn api:app --port $PORT --host 0.0.0.0 --workers 1
