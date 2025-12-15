# Use a base image with Python and CUDA support (if using GPU)
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install Python and essential packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY handler.py .
# Note: You don't need to copy the Judgments/Summaries directories for the API endpoint
# Command to run when the container starts
CMD ["python3", "-u", "handler.py"]

