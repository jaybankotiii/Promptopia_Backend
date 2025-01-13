#!/usr/bin/env bash

echo "🔄 Installing Node.js dependencies..."
npm install

echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

echo "📦 Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt
