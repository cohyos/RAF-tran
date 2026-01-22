# RAF-Tran Docker Image
# Multi-stage build for optimized image size

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir .

# Stage 2: Runtime image
FROM python:3.11-slim as runtime

LABEL maintainer="RAF-Tran Contributors"
LABEL description="Open-source atmospheric radiative transfer simulation"
LABEL version="0.1.0"

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY raf_tran/ ./raf_tran/

# Create data directory for volume mount
RUN mkdir -p /app/data/spectral_db /app/output

# Set environment variables for offline mode
ENV RAF_TRAN_OFFLINE_MODE=True
ENV RADIS_OFFLINE_MODE=True
ENV PYTHONUNBUFFERED=1

# Volume mount points
# - /app/data: Mount external data directory containing spectral database
# - /app/output: Mount for output files
VOLUME ["/app/data", "/app/output"]

# Default command
ENTRYPOINT ["python", "-m", "raf_tran.cli"]
CMD ["--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import raf_tran; print(raf_tran.__version__)" || exit 1
