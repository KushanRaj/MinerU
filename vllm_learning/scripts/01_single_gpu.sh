#!/bin/bash
# =============================================================================
# Script: 01_single_gpu.sh
# Purpose: Basic single GPU vLLM server for learning
# =============================================================================

set -e  # Exit on error

echo "🚀 Starting Single GPU vLLM Server"
echo "===================================="
echo ""
echo "📊 Configuration:"
echo "  • GPU: 0 (single GPU)"
echo "  • Port: 30000"
echo "  • Memory Utilization: 80%"
echo "  • Max Sequences: 4"
echo "  • Max Model Length: 4096 tokens"
echo ""
echo "🎓 Learning Focus:"
echo "  • Understand basic vLLM parameters"
echo "  • Observe GPU memory usage"
echo "  • Monitor request processing"
echo ""
echo "📝 Monitor this server:"
echo "  • GPU usage: watch -n 1 nvidia-smi"
echo "  • Test: mineru -p <pdf> -o outputs/test -b vlm-http-client -u http://localhost:30000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="
echo ""

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Set environment
export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0
export VLLM_LOGGING_LEVEL=INFO

# Create logs directory if it doesn't exist
mkdir -p logs

# Start server
mineru-vllm-server \
  --host 0.0.0.0 \
  --port 30000 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 4 \
  --max-model-len 4096 \
  2>&1 | tee logs/single_gpu.log

nohup mineru-vllm-server \
  --host 0.0.0.0 \
  --port 30000 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 4 \
  --max-model-len 4096 \
  < /dev/null > logs/single_gpu.log 2>&1 &


