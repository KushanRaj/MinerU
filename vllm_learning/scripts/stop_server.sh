#!/bin/bash
# =============================================================================
# Script: stop_server.sh
# Purpose: Stop all running vLLM servers
# =============================================================================

echo "🛑 Stopping all vLLM servers..."
echo ""

# Find and kill mineru-vllm-server processes
PIDS=$(pgrep -f "mineru-vllm-server" || true)

if [ -z "$PIDS" ]; then
    echo "✅ No vLLM servers running"
else
    echo "📋 Found vLLM server processes:"
    ps aux | grep mineru-vllm-server | grep -v grep || true
    echo ""
    echo "🔪 Killing processes: $PIDS"
    kill $PIDS 2>/dev/null || sudo kill $PIDS
    
    # Wait a bit and force kill if necessary
    sleep 2
    REMAINING=$(pgrep -f "mineru-vllm-server" || true)
    if [ -n "$REMAINING" ]; then
        echo "⚠️  Force killing remaining processes: $REMAINING"
        kill -9 $REMAINING 2>/dev/null || sudo kill -9 $REMAINING
    fi
    
    echo "✅ All vLLM servers stopped"
fi

# Check if port 30000 is still in use
if lsof -i :30000 >/dev/null 2>&1; then
    echo ""
    echo "⚠️  Port 30000 is still in use:"
    lsof -i :30000
    echo ""
    echo "Run this to kill: sudo lsof -ti :30000 | xargs kill -9"
else
    echo "✅ Port 30000 is free"
fi

echo ""
echo "Done!"



