#!/usr/bin/env bash

# Install Node.js dependencies
npm install

# Set up Python virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install --no-cache-dir -r requirements.txt