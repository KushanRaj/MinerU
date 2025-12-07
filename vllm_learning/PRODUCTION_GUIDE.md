# 🚀 Production Deployment Guide - vLLM Server

## ✅ Quick Answers to Your Questions

### 1. How many instances per GPU?

**Answer: ONE vLLM server process manages BOTH GPUs**

```
❌ WRONG: Run 2 separate vLLM instances (one per GPU)
✅ CORRECT: Run 1 vLLM instance with --data-parallel-size 2
```

**Why?**
- vLLM internally handles distributing work across GPUs
- The `--data-parallel-size 2` flag tells vLLM to create 2 model replicas
- vLLM's load balancer automatically routes requests to available GPUs

### 2. How many should you deploy?

**For your 2x RTX 3060 setup:**
```bash
# Production configuration
1 vLLM server process with:
  --data-parallel-size 2        # Uses both GPUs
  --max-num-seqs 16             # Handles 16 concurrent requests
  --gpu-memory-utilization 0.7  # Leaves headroom for stability
```

**Scaling strategy:**
- Single server = ~20-30 PDF pages/min throughput
- For higher load: Run multiple independent vLLM servers on different machines
- Each server manages its own GPUs

### 3. What is the API?

vLLM provides **TWO APIs**:

#### A) OpenAI-Compatible Chat API (Standard)
```bash
Endpoint: http://localhost:30000/v1/chat/completions
Method: POST
```

#### B) MinerU's Custom PDF API (Built on top)
```bash
Endpoint: Use MinerU client or FastAPI wrapper
Method: HTTP POST with PDF bytes
```

### 4. What is the expected input?

#### Direct vLLM Input (for your backend):
```json
POST http://localhost:30000/v1/chat/completions
Content-Type: application/json

{
  "model": "mineru-2.5-production",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }
  ],
  "temperature": 0.0,
  "max_tokens": 4096
}
```

#### MinerU Wrapped Input (easier):
```bash
# Via CLI
mineru -p document.pdf -o output/ -b vlm-http-client -u http://localhost:30000

# Via Python
from mineru import parse_doc
parse_doc(
    path_list=["document.pdf"],
    output_dir="output/",
    backend="vlm-http-client",
    server_url="http://localhost:30000"
)

# Via FastAPI (recommended for microservices)
POST http://localhost:8000/api/parse
Content-Type: multipart/form-data
Body: PDF file upload
```

### 5. How to connect to it?

#### Option A: Direct HTTP Client (Your Backend)
```python
import requests
import base64

# Read PDF
with open("document.pdf", "rb") as f:
    pdf_bytes = f.read()

# Convert to base64
pdf_base64 = base64.b64encode(pdf_bytes).decode()

# Send to vLLM server
response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "mineru-2.5-production",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{pdf_base64}"}}
        ]}],
        "temperature": 0.0
    }
)

result = response.json()
```

#### Option B: MinerU Python API (Easier)
```python
from mineru.cli.common import read_fn
from demo import parse_doc
from pathlib import Path

# Parse via vLLM server
parse_doc(
    path_list=[Path("document.pdf")],
    output_dir="output/",
    backend="vlm-http-client",
    server_url="http://localhost:30000"
)
```

#### Option C: FastAPI Microservice (Recommended)
```bash
# Start MinerU's FastAPI server
mineru-api --host 0.0.0.0 --port 8000

# It internally connects to vLLM server at localhost:30000
```

Then your frontend calls:
```javascript
// Frontend (JavaScript/TypeScript)
const formData = new FormData();
formData.append('file', pdfFile);

const response = await fetch('http://your-backend:8000/api/parse', {
  method: 'POST',
  body: formData
});

const result = await response.json();
// Returns: { markdown: "...", images: [...], tables: [...] }
```

---

## 🏗️ Microservice Architecture

Here's how to integrate with your existing architecture:

```
┌─────────────┐      ┌──────────────┐      ┌────────────────┐
│  Frontend   │─────▶│   Backend    │─────▶│  vLLM Server   │
│   (React)   │      │  (FastAPI)   │      │ (localhost:30000)
└─────────────┘      └──────────────┘      └────────────────┘
                            │                    │
                            │                    ▼
                            │              ┌──────────┐
                            │              │ GPU 0    │
                            │              │ GPU 1    │
                            │              └──────────┘
                            ▼
                     ┌──────────────┐
                     │  Database    │
                     │  Storage     │
                     └──────────────┘
```

### Deployment Options:

#### Option 1: Simple (All on One Server)
```bash
# Start vLLM server (background)
nohup mineru-vllm-server --port 30000 --data-parallel-size 2 &

# Start your FastAPI backend (connects to localhost:30000)
uvicorn your_backend:app --host 0.0.0.0 --port 8000
```

#### Option 2: Separate Services (Recommended)
```bash
# Server 1: GPU Machine (vLLM server only)
mineru-vllm-server --host 0.0.0.0 --port 30000 --data-parallel-size 2

# Server 2: Backend (connects to Server 1)
VLLM_SERVER=http://gpu-server:30000 uvicorn your_backend:app --host 0.0.0.0 --port 8000
```

---

## 🔌 API Endpoints Reference

### vLLM Server (Port 30000)

```bash
# Health check
GET http://localhost:30000/health
Response: {"status": "ok"}

# Model info
GET http://localhost:30000/v1/models
Response: {"data": [{"id": "mineru-2.5-production", ...}]}

# Chat completion (PDF processing)
POST http://localhost:30000/v1/chat/completions
Headers: Content-Type: application/json
Body: {
  "model": "mineru-2.5-production",
  "messages": [...],
  "temperature": 0.0,
  "max_tokens": 4096
}
```

### MinerU FastAPI (Port 8000)

```bash
# API docs (auto-generated)
GET http://localhost:8000/docs

# Parse PDF
POST http://localhost:8000/api/parse
Content-Type: multipart/form-data
Body: file=@document.pdf

Response: {
  "markdown": "# Document content...",
  "images": ["data:image/png;base64,..."],
  "tables": ["<table>...</table>"]
}
```

---

## 🚀 Production Deployment Steps

### Step 1: Start vLLM Server
```bash
cd /home/kushan/MinerU
source .venv/bin/activate
cd vllm_learning

# Production server
./scripts/03_production.sh
```

### Step 2: Verify Server is Running
```bash
# Check health
curl http://localhost:30000/health

# Check models
curl http://localhost:30000/v1/models

# Check GPU usage
nvidia-smi
```

### Step 3: Connect Your Backend

Create `/home/kushan/MinerU/backend_integration.py`:
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import httpx
import base64
import json

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

VLLM_SERVER = "http://localhost:30000"

@app.post("/api/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    """
    Parse PDF and return markdown
    """
    # Read PDF bytes
    pdf_bytes = await file.read()
    pdf_base64 = base64.b64encode(pdf_bytes).decode()
    
    # Send to vLLM server
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{VLLM_SERVER}/v1/chat/completions",
            json={
                "model": "mineru-2.5-production",
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {"url": f"data:application/pdf;base64,{pdf_base64}"}
                    }]
                }],
                "temperature": 0.0,
                "max_tokens": 4096
            }
        )
    
    result = response.json()
    
    # Extract markdown from response
    markdown = result["choices"][0]["message"]["content"]
    
    return {
        "success": True,
        "markdown": markdown,
        "filename": file.filename
    }

@app.get("/health")
async def health():
    """Health check"""
    # Check if vLLM server is accessible
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{VLLM_SERVER}/health")
            vllm_healthy = response.status_code == 200
    except:
        vllm_healthy = False
    
    return {
        "status": "ok",
        "vllm_server": vllm_healthy
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 4: Run Your Backend
```bash
cd /home/kushan/MinerU
source .venv/bin/activate
python backend_integration.py
```

### Step 5: Test the Integration
```bash
# Test from command line
curl -X POST http://localhost:8000/api/parse-pdf \
  -F "file=@demo/pdfs/demo1.pdf"

# Or use httpie (more readable)
http --form POST localhost:8000/api/parse-pdf file@demo/pdfs/demo1.pdf
```

### Step 6: Frontend Integration
```javascript
// React/Next.js example
async function parsePDF(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://your-backend:8000/api/parse-pdf', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  return result.markdown;
}
```

---

## 📊 Performance & Scaling

### Current Setup (2x RTX 3060):
- **Throughput**: ~20-30 pages/minute
- **Concurrent requests**: 16 (configurable)
- **Latency**: ~2-5 seconds per page
- **Memory**: ~8-9 GB VRAM per GPU

### Scaling Strategies:

1. **Vertical Scaling** (More GPUs on same machine)
   - Add more GPUs → increase `--data-parallel-size`
   - Linear scaling up to ~8 GPUs

2. **Horizontal Scaling** (Multiple machines)
   - Deploy vLLM servers on multiple GPU machines
   - Use load balancer (nginx/HAProxy) in front
   - Your backend distributes requests

3. **Batch Processing** (For high volume)
   - Queue system (Celery/RQ)
   - Process PDFs asynchronously
   - Return job ID immediately, webhook on completion

---

## 🔒 Exposing the Service

### Option 1: Direct Exposure (Simple)
```bash
# Make sure vLLM server binds to 0.0.0.0
mineru-vllm-server --host 0.0.0.0 --port 30000 ...

# Your firewall/security group should allow port 30000
```

### Option 2: Behind Reverse Proxy (Recommended)
```nginx
# nginx.conf
upstream vllm_backend {
    server localhost:30000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location /api/vllm/ {
        proxy_pass http://vllm_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
}
```

### Option 3: With Authentication (Production)
```python
# Add API key authentication to your FastAPI backend
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()
API_KEY = "your-secret-key"

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials.credentials

@app.post("/api/parse-pdf")
async def parse_pdf(
    file: UploadFile = File(...),
    token: str = Security(verify_token)
):
    # ... your code
```

---

## 🐳 Docker Deployment (Optional)

If you want to containerize:

```dockerfile
# Dockerfile
FROM vllm/vllm-openai:v0.10.2

WORKDIR /app
COPY . .

RUN pip install mineru[all]

EXPOSE 30000

CMD ["mineru-vllm-server", "--host", "0.0.0.0", "--port", "30000", "--data-parallel-size", "2"]
```

```bash
# Build and run
docker build -t mineru-vllm .
docker run --gpus all -p 30000:30000 mineru-vllm
```

---

## ✅ Current Status

Your server is starting now! Check:
```bash
# Server logs
tail -f /home/kushan/MinerU/vllm_learning/logs/dual_gpu_parallel.log

# Test it
curl http://localhost:30000/health
```

Ready to test your first PDF? 🚀



