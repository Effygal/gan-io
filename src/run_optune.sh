#!/bin/bash
# =============================================================================
# Local Development Optimization Script
# =============================================================================
# 
# USAGE:
#   ./run_optune.sh <trace_name> [device] [n_trials]
# 
# EXAMPLES:
#   ./run_optune.sh v521 mps 50        # Run v521 on Mac M1 with 50 trials
#   ./run_optune.sh w44 cuda 30        # Run w44 on CUDA with 30 trials
# 
# AVAILABLE TRACES:
#   v521, v766, v827, v538, w44, w82, w24, w11
# 
=============================================================================

if [ $# -lt 1 ]; then
    echo "Usage: $0 <trace_name> [device] [n_trials]"
    echo "Available traces: v521, v766, v827, v538, w44, w82, w24, w11"
    echo "Example: $0 v521 mps 50"
    exit 1
fi

TRACE_NAME=$1
DEVICE=${2:-"cuda"}  # mps for Mac M1, cuda for HPC
N_TRIALS=${3:-50} 

echo "Starting Optuna optimization for trace: $TRACE_NAME"
echo "Device: $DEVICE"
echo "Trials: $N_TRIALS"
echo ""

# Run the optimization
python optune.py \
    --trace-name "$TRACE_NAME" \
    --device "$DEVICE" \
    --n-trials "$N_TRIALS"

echo ""
echo "Optimization completed for $TRACE_NAME"
