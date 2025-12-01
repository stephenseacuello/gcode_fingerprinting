#!/usr/bin/env bash
################################################################################
# start.sh - Setup virtual environment and start Flask application
################################################################################
#
# This script:
# 1. Creates a Python virtual environment if it doesn't exist
# 2. Activates the virtual environment
# 3. Upgrades pip
# 4. Installs dependencies from requirements.txt
# 5. Launches the Flask application
#
# Usage:
#   ./scripts/start.sh
#
################################################################################

set -e  # Exit on error

# Get project root directory (assuming scripts/ is in project root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "════════════════════════════════════════════════════════════════════"
echo "  Flask Application Startup"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt --quiet

echo ""
echo "✓ Setup complete"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Starting Flask Application"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Check for Flask app locations
if [ -f "src/miracle/app/app.py" ]; then
    APP_PATH="src/miracle/app/app.py"
elif [ -f "src/miracle/app/flask_cnc_controller.py" ]; then
    APP_PATH="src/miracle/app/flask_cnc_controller.py"
elif [ -f "app.py" ]; then
    APP_PATH="app.py"
else
    echo "ERROR: Could not find Flask application"
    echo "Searched for:"
    echo "  - src/miracle/app/app.py"
    echo "  - src/miracle/app/flask_cnc_controller.py"
    echo "  - app.py"
    exit 1
fi

echo "Launching: $APP_PATH"
echo ""

python "$APP_PATH"
