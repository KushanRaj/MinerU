#!/bin/bash
# =============================================================================
# Script: 02_dual_gpu_parallel.sh
# Purpose: Dual GPU data parallel vLLM server (RECOMMENDED)
# =============================================================================

set -e  # Exit on error

echo "🚀 Starting Dual GPU Data Parallel vLLM Server"
echo "==============================================="
echo ""
echo "📊 Configuration:"
echo "  • GPUs: 0 & 1 (data parallel)"
echo "  • Port: 30000"
echo "  • Memory Utilization: 70% per GPU"
echo "  • Max Sequences: 16 (distributed across GPUs)"
echo "  • Max Model Length: 4096 tokens"
echo ""
echo "🎓 Learning Focus:"
echo "  • Data parallelism - 2 model instances"
echo "  • Load balancing across GPUs"
echo "  • Higher throughput vs single GPU"
echo ""
echo "⚡ Expected Performance:"
echo "  • ~2x throughput compared to single GPU"
echo "  • Both GPUs should show balanced load"
echo ""
echo "📝 Monitor this server:"
echo "  • Both GPUs: watch -n 1 nvidia-smi"
echo "  • Test: mineru -p <pdf> -o outputs/test -b vlm-http-client -u http://localhost:30000"
echo "  • Benchmark: python benchmarks/benchmark_server.py"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================================"
echo ""

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Set environment
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
export VLLM_LOGGING_LEVEL=INFO

# Create logs directory if it doesn't exist
mkdir -p logs

# Start server with data parallelism
mineru-vllm-server \
  --host 0.0.0.0 \
  --port 30000 \
  --data-parallel-size 2 \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 16 \
  --max-model-len 4096 \
  --enable-chunked-prefill \
  2>&1 | tee logs/dual_gpu_parallel.log



