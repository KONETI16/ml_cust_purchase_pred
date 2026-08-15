#1. Lightweight python image for runtime
FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Ensure environment is non-buffered (good for logging)
ENV PYTHONUNBUFFERED=1

# Default to the host MLflow tracking server used during local Docker Desktop runs.
# Override this at runtime if your tracking server is elsewhere.
ENV MLFLOW_TRACKING_URI="http://host.docker.internal:5000"

# Copy only requirements first to leverage Docker cache
COPY requirements.txt ./

# Install OS-level build dependencies required for scientific packages, upgrade pip, then install requirements
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
	 && apt-get install -y --no-install-recommends \
		 build-essential \
		 gcc \
		 g++ \
		 gfortran \
		 pkg-config \
		 libopenblas-dev \
		 liblapack-dev \
		 libpng-dev \
		 libfreetype6-dev \
		 libjpeg-dev \
		 zlib1g-dev \
		 ca-certificates \
	&& rm -rf /var/lib/apt/lists/* \
	&& pip install --upgrade pip setuptools wheel \
	&& pip install --no-cache-dir -r requirements.txt

# Copy application files (including pipeline.pkl)
COPY . .

# Expose the port that the application will run on
EXPOSE 8000

# Default command to run the FastAPI app via uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]