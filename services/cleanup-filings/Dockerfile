FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY main.py /app/main.py

ENV PYTHONUNBUFFERED=1
CMD exec gunicorn -b :${PORT} --workers 1 --threads 8 --timeout 0 main:app
