# 🚀 vLLM Learning Environment

Welcome to your hands-on vLLM learning setup! This environment is designed to help you master model serving with vLLM using your 2x RTX 3060 GPUs.

## 📁 Directory Structure

```
vllm_learning/
├── README.md                 # This file
├── scripts/                  # Server startup scripts
│   ├── 01_single_gpu.sh     # Basic single GPU server
│   ├── 02_dual_gpu_parallel.sh  # Dual GPU data parallel (RECOMMENDED)
│   ├── 03_production.sh     # Production-grade configuration
│   └── stop_server.sh       # Stop all servers
├── configs/                  # Configuration files
│   └── vllm_params.txt      # vLLM parameter reference
├── benchmarks/               # Performance testing
│   ├── benchmark_server.py  # Automated benchmarking
│   └── compare_configs.sh   # Compare single vs dual GPU
├── monitoring/               # Monitoring tools
│   ├── monitor_gpus.py      # Real-time GPU monitoring
│   └── server_logs.sh       # View server logs
├── outputs/                  # Parsed PDF outputs
├── logs/                     # Server logs
└── test_pdfs/               # Test PDFs (symlinked from demo)
```

## 🎯 Your Hardware Setup

- **GPUs**: 2x NVIDIA GeForce RTX 3060 (11.6 GB VRAM each)
- **Driver**: 570.195.03
- **CUDA**: 12.8
- **Total VRAM**: ~23 GB

## 🚦 Quick Start

### Phase 1: Basic Single GPU (Learning)
```bash
# Activate environment
cd /home/kushan/MinerU && source .venv/bin/activate

# Start single GPU server
cd vllm_learning
./scripts/01_single_gpu.sh

# In another terminal, test it
mineru -p ../demo/pdfs/demo1.pdf -o outputs/test1 -b vlm-http-client -u http://localhost:30000
```

### Phase 2: Dual GPU Data Parallel (Recommended)
```bash
# Start dual GPU server
./scripts/02_dual_gpu_parallel.sh

# Monitor GPUs in another terminal
./monitoring/monitor_gpus.py

# Benchmark performance
python benchmarks/benchmark_server.py
```

### Phase 3: Production Deployment
```bash
# Production config with logging
./scripts/03_production.sh

# View logs
./monitoring/server_logs.sh
```

## 📚 Learning Path

### Week 1: Fundamentals
- [x] Installation complete
- [x] Environment setup
- [ ] Run single GPU server
- [ ] Understand vLLM parameters
- [ ] Monitor GPU usage

### Week 2: Optimization
- [ ] Deploy dual GPU server
- [ ] Compare single vs dual GPU performance
- [ ] Run benchmarks
- [ ] Optimize parameters for your workload

### Week 3: Production
- [ ] Production configuration
- [ ] Error handling & logging
- [ ] Load testing
- [ ] Deploy as systemd service (optional)

## 🔑 Key Concepts

### Data Parallelism (What We're Using)
```
GPU 0: Full Model Instance (11.6GB) → Process Request A, C, E...
GPU 1: Full Model Instance (11.6GB) → Process Request B, D, F...
```
- **Advantage**: 2x throughput, no communication overhead
- **Best for**: Small models (1.2B) that fit on one GPU

### Important vLLM Parameters

| Parameter | Description | Your Setup |
|-----------|-------------|------------|
| `--data-parallel-size 2` | Use both GPUs | ✅ Use this |
| `--gpu-memory-utilization 0.7` | Use 70% of VRAM for KV cache | Start here |
| `--max-num-seqs 16` | Max concurrent requests | Tune based on workload |
| `--max-model-len 4096` | Max tokens per request | Adjust if needed |

## 🎓 Learning Exercises

1. **GPU Utilization**: Start the server and watch both GPUs with `nvidia-smi`
2. **Load Balancing**: Send multiple requests and see how they're distributed
3. **Memory Management**: Adjust `--gpu-memory-utilization` and observe
4. **Throughput Testing**: Run benchmarks with different concurrency levels

## 🔧 Troubleshooting

### Server won't start
```bash
# Check if port is in use
lsof -i :30000

# Kill existing server
./scripts/stop_server.sh
```

### Out of Memory
```bash
# Reduce memory utilization
--gpu-memory-utilization 0.5

# Reduce concurrent sequences
--max-num-seqs 8
```

### One GPU not being used
```bash
# Verify CUDA sees both GPUs
python -c "import torch; print(torch.cuda.device_count())"

# Check server logs
tail -f logs/server.log
```

## 📊 Expected Performance

Based on MinerU 2.5 (1.2B model) on 2x RTX 3060:

| Configuration | Throughput | Best For |
|---------------|------------|----------|
| Single GPU | ~10 pages/min | Learning, testing |
| Dual GPU (Data Parallel) | ~18-20 pages/min | Production |
| vLLM vs Transformers | 20-30x faster | Always use vLLM |

## 🎯 Next Steps

1. Read through all the scripts in `scripts/`
2. Run `01_single_gpu.sh` and understand the output
3. Compare with `02_dual_gpu_parallel.sh`
4. Run benchmarks and analyze results
5. Experiment with parameters

---

**Ready to start? Run your first server:**
```bash
cd /home/kushan/MinerU && source .venv/bin/activate
cd vllm_learning && ./scripts/01_single_gpu.sh
```



