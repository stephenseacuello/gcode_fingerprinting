#!/usr/bin/env bash
################################################################################
# launch_mcc_helper.sh - Launch MCC DAQ Helper for 4-Channel Current Monitoring
################################################################################
#
# This script launches the MCC helper under Rosetta Intel Python (x86_64)
# with the correct environment and library paths.
#
# Hardware Configuration:
#   Channel 0: Spindle motor current
#   Channel 1: X-axis stepper motor current
#   Channel 2: Y-axis stepper motor current
#   Channel 3: Z-axis stepper motor current
#
# Usage:
#   ./scripts/launch_mcc_helper.sh                           # Default settings
#   ./scripts/launch_mcc_helper.sh tcp://127.0.0.1:5558      # Custom port
#   ./scripts/launch_mcc_helper.sh tcp://127.0.0.1:5558 2000 # Custom port + rate
#   ./scripts/launch_mcc_helper.sh tcp://127.0.0.1:5558 2000 100  # Full custom
#
################################################################################

# Intel/x86_64 conda environment path
INTEL_ENV="$HOME/miniforge-x86_64/envs/mcc-intel"
INTEL_PY="$INTEL_ENV/bin/python"

# Make sure the Intel env's lib dir is visible for libuldaq.dylib
export DYLD_FALLBACK_LIBRARY_PATH="$INTEL_ENV/lib:${DYLD_FALLBACK_LIBRARY_PATH}"

# Configuration (can be overridden by command-line arguments)
PUB_URL="${1:-tcp://127.0.0.1:5557}"   # ZeroMQ publish URL
RATE="${2:-1000}"                       # Sample rate per channel (Hz)
BLOCK="${3:-200}"                       # Samples per channel per block

# 4-channel configuration for current sensors
CH_LOW=0
CH_HIGH=3

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║       MCC DAQ Helper - 4-Channel Current Monitor Launcher          ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Display configuration
echo -e "${BLUE}Configuration:${NC}"
echo "  ZeroMQ URL:    $PUB_URL"
echo "  Sample Rate:   $RATE Hz per channel"
echo "  Block Size:    $BLOCK samples per channel"
echo "  Channels:      $CH_LOW to $CH_HIGH"
echo ""
echo "  Channel Mapping:"
echo "    CH0 → Spindle motor current"
echo "    CH1 → X-axis stepper motor current"
echo "    CH2 → Y-axis stepper motor current"
echo "    CH3 → Z-axis stepper motor current"
echo ""

# Verify Intel Python exists
if [ ! -f "$INTEL_PY" ]; then
    echo -e "${YELLOW}ERROR: Intel Python not found at $INTEL_PY${NC}"
    echo ""
    echo "Please ensure you have created the mcc-intel conda environment:"
    echo "  conda create -n mcc-intel python=3.9"
    echo "  conda activate mcc-intel"
    echo "  pip install uldaq pyzmq"
    exit 1
fi

# Verify architecture
echo -e "${BLUE}Verifying environment...${NC}"
arch -x86_64 "$INTEL_PY" -c 'import platform; print(f"  Python arch: {platform.machine()}")' || {
    echo -e "${YELLOW}ERROR: Failed to run Python in x86_64 mode${NC}"
    exit 1
}

# Check required packages
arch -x86_64 "$INTEL_PY" -c 'import uldaq; import zmq' 2>/dev/null || {
    echo -e "${YELLOW}ERROR: Required packages not installed${NC}"
    echo ""
    echo "Install with:"
    echo "  conda activate mcc-intel"
    echo "  pip install uldaq pyzmq"
    exit 1
}

echo -e "${GREEN}  ✓ Environment verified${NC}"
echo ""

# Instructions
echo -e "${BLUE}Instructions:${NC}"
echo "  1. Ensure USB-1608G is connected to Mac host (not VM)"
echo "  2. In Flask app, use 'Discover MCC Ports' and select: $PUB_URL"
echo "  3. Click 'Connect MCC' in the web interface"
echo "  4. Start recording when ready"
echo ""
echo "  Press Ctrl+C to stop the helper"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Launch the helper
echo -e "${GREEN}Starting MCC DAQ Helper...${NC}"
echo ""

# Get project root directory (assuming scripts/ is in project root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Path to mcc_helper.py in reorganized structure
MCC_HELPER="$PROJECT_ROOT/src/miracle/utilities/mcc_helper.py"

if [ ! -f "$MCC_HELPER" ]; then
    echo -e "${YELLOW}ERROR: mcc_helper.py not found at $MCC_HELPER${NC}"
    echo "Please ensure the file exists at: src/miracle/utilities/mcc_helper.py"
    exit 1
fi

exec arch -x86_64 "$INTEL_PY" "$MCC_HELPER" \
    --pub "$PUB_URL" \
    --rate "$RATE" \
    --block "$BLOCK" \
    --ch-low "$CH_LOW" \
    --ch-high "$CH_HIGH"
