# Base image
FROM python:3.12-slim

EXPOSE $PORT

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY requirements_backend.txt /app/requirements_backend.txt
COPY src/my_project/backend.py /app/backend.py
COPY models/model.onnx models/model.onnx
COPY src/my_project/image_analysis.py /app/image_analysis.py

RUN pip install -r requirements_backend.txt --no-cache-dir

CMD exec uvicorn backend:app --port $PORT --host 0.0.0.0
