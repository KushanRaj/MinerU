#!/bin/bash
# =============================================================================
# Script: 03_production.sh
# Purpose: Production-grade vLLM server with optimal settings
# =============================================================================

set -e  # Exit on error

echo "🚀 Starting Production vLLM Server"
echo "===================================="
echo ""
echo "📊 Configuration:"
echo "  • GPUs: 0 & 1 (data parallel)"
echo "  • Port: 30000"
echo "  • Memory Utilization: 75%"
echo "  • Max Sequences: 16"
echo "  • Chunked Prefill: Enabled"
echo "  • Logging: Full (saved to logs/)"
echo ""
echo "⚙️ Production Features:"
echo "  • Optimized memory settings"
echo "  • Comprehensive logging"
echo "  • Named model endpoint"
echo "  • Performance monitoring"
echo ""
echo "📝 Access points:"
echo "  • Server: http://localhost:30000"
echo "  • Logs: tail -f logs/production.log"
echo "  • Monitor: ./monitoring/monitor_gpus.py"
echo ""
echo "Press Ctrl+C to stop the server"
echo "===================================="
echo ""

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Set environment
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_LOGGING_LEVEL=INFO

# Create logs and outputs directories
mkdir -p logs outputs

# Get timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "📝 Starting server at $(date)"
echo "📁 Logs will be saved to: logs/production.log"
echo "📁 Timestamped log: logs/production_${TIMESTAMP}.log"
echo ""

# Start production server
mineru-vllm-server \
  --host 0.0.0.0 \
  --port 30000 \
  --data-parallel-size 2 \
  --gpu-memory-utilization 0.75 \
  --max-num-seqs 16 \
  --max-model-len 4096 \
  --enable-chunked-prefill \
  --served-model-name mineru-2.5-production \
  2>&1 | tee logs/production.log logs/production_${TIMESTAMP}.log

echo ""
echo "🛑 Server stopped at $(date)"



