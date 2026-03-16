FROM python:3.11

WORKDIR /app

# create this file with poetry export
COPY requirements.txt ./
RUN pip install --no-cache-dir --no-deps -r requirements.txt

# Copy only the project files we need
COPY src/ ./src
COPY artifacts/training/ ./artifacts/training/
COPY README.md ./

# model path is different within the container
ENV MODEL_PATH=artifacts/training
ENV PYTHONUNBUFFERED=1

# Expose Flask API default port
EXPOSE 80

# Run the API
CMD ["python", "-m", "src.app","--host", "0.0.0.0", "--port", "80"]