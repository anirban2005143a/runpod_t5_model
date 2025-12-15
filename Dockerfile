# 1. BASE IMAGE: Use a Python/CUDA runtime image from NVIDIA or PyTorch.
#    The runtime image is smaller than the devel (development) image.
#    We use a PyTorch-provided image as it guarantees PyTorch is compatible 
#    with the included CUDA/cuDNN versions. Adjust the tag for your needs.
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# 2. METADATA: Set environment variable for the port your application will listen on
#    (FastAPI/Uvicorn default is often 8000). The serverless platform needs this.
ENV PORT 8000
EXPOSE 8000

# 3. SETUP: Set the working directory inside the container
WORKDIR /app

# 4. DEPENDENCIES: Copy requirements.txt first to leverage Docker layer caching.
#    If requirements.txt doesn't change, Docker won't rerun this time-consuming step.
COPY requirements.txt .

# 5. INSTALL: Install Python dependencies. 
#    --no-cache-dir saves space. -r installs everything from the list.
RUN pip install --no-cache-dir -r requirements.txt

# 6. APPLICATION CODE: Copy the rest of your local repository files into the container.
#    This includes app.py, your custom chunking modules, configs, etc.
COPY . .

# 7. MODEL LOADING (Optional but recommended for faster cold starts):
#    If your model is small or you have access to a persistent cache/volume,
#    pre-downloading it here can reduce cold start latency significantly.
#    Example: RUN python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('your-llm-model-name'); AutoTokenizer.from_pretrained('your-llm-model-name')"

# 8. STARTUP COMMAND: Run your API server.
#    - Uvicorn is a common ASGI server for high-performance Python APIs (FastAPI).
#    - `app.py` is your main script, and `app` is the FastAPI application object inside it.
#    - Bind to 0.0.0.0 for external access (required by serverless platforms).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]