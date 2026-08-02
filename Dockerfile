FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .

# Install CPU-only PyTorch separately -- the default PyPI package includes
# ~900MB of CUDA/GPU libraries that are never used on this instance and
# were confirmed (by an out-of-memory crash) to contribute to exceeding
# Render's 512MB free-tier limit at runtime.
RUN pip install --no-cache-dir torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
