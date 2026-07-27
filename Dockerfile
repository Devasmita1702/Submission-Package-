FROM python:3.10-slim

# Install geospatial dependencies (GDAL/PROJ C-libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source solution code
COPY solution/ ./solution/

ENV PYTHONPATH="/app"

ENTRYPOINT ["python", "-m", "solution.infer"]
