# # Gunakan Python base image
# FROM python:3.11-slim

# # Set working directory
# WORKDIR /app

# # Install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy semua file ke container
# COPY . .

# # Jalankan FastAPI dengan Uvicorn
# # CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["python", "main.py"]


# # docker build -t rekomendasi-app .
# # docker run -d -p 8000:8080 --name rekomendasi-container rekomendasi-app


# Gunakan base image Python yang ringan
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install OS-level dependencies (opsional jika error runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential gcc && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Salin file requirements dan install dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file source code ke dalam image
COPY . .

# Jalankan FastAPI (port 8080 karena disesuaikan dengan main.py)
CMD ["python", "main.py"]

