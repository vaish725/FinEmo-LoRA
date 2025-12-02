#!/bin/bash

# FinEmo-LoRA Dashboard Launch Script

echo "================================"
echo "  FinEmo-LoRA Dashboard Launcher"
echo "================================"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found!"
    echo "Installing dependencies..."
    pip install -r ../requirements_app.txt
fi

# Check if model exists
if [ ! -d "../models/finemo-lora-final-v2" ]; then
    echo "⚠️  Warning: Model not found at ../models/finemo-lora-final-v2"
    echo "Please ensure the LoRA v2 model is downloaded."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✅ Launching dashboard..."
echo ""
echo "📊 Dashboard will open at: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

# Launch Streamlit
streamlit run app.py
