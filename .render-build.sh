#!/usr/bin/env bash
# Install Python dependencies in a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --no-cache-dir -r requirements.txt