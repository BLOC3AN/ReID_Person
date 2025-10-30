#!/bin/bash
# Launch Streamlit UI for Person ReID System

# Activate virtual environment if exists
if [ -d "../hai_venv" ]; then
    source ../hai_venv/bin/activate
    echo "✅ Activated hai_venv"
fi

# Launch Streamlit
echo "🚀 Starting Person ReID UI..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

