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

# RunPod expects a command to start the worker. 
# The base RunPod Serverless template uses a handler script.
# We'll use the official RunPod worker base for simplicity in a real setup, 
# but for a custom script, you'd typically run it like this:
# CMD ["python3", "handler.py"] 
# However, for a RunPod worker, the entrypoint is usually defined by the 
# worker environment. For this example, we'll assume a standard RunPod worker setup 
# where 'handler.py' is the entrypoint.
