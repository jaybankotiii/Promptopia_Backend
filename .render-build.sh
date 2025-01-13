#!/usr/bin/env bash

echo "🔄 Installing Node.js dependencies..."
npm install

echo "🐍 Setting up Python environment..."
echo "🔧 Creating Python virtual environment..."
python3 -m venv venv

echo "📦 Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

echo "✅ Python environment setup completed!"
