FROM python:3.12-slim

WORKDIR /app

RUN pip install fastapi evidently numpy pandas google-cloud-storage --no-cache-dir

COPY src/my_project/app_monitoring.py /app/app_monitoring.py
COPY src/my_project/image_analysis.py /app/image_analysis.py

EXPOSE $PORT

CMD exec uvicorn app_monitoring:app --port $PORT --host 0.0.0.0
