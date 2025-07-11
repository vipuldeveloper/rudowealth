#!/bin/bash
# Build script for Vercel deployment

# Install system dependencies for FAISS
apt-get update
apt-get install -y build-essential

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

echo "Build completed successfully" 