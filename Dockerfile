# RAF-tran Docker Image
# =====================
#
# Build:
#   docker build -t raf-tran .
#
# Run (CLI):
#   docker run -it raf-tran python examples/01_solar_zenith_angle_study.py --no-plot
#
# Run (Streamlit GUI):
#   docker run -p 8501:8501 raf-tran streamlit run streamlit_app/app.py --server.address 0.0.0.0
#
# Interactive shell:
#   docker run -it raf-tran bash

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Streamlit for GUI
RUN pip install --no-cache-dir streamlit pandas

# Copy the rest of the application
COPY . .

# Install RAF-tran in development mode
RUN pip install -e .

# Create outputs directory
RUN mkdir -p outputs

# Default command - show help
CMD ["python", "-c", "import raf_tran; print('RAF-tran', raf_tran.__version__, '- Atmospheric Radiative Transfer Library')"]

# Expose Streamlit port
EXPOSE 8501
