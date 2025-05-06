FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY qa_service.py .

ENTRYPOINT ["python", "qa_service.py"]