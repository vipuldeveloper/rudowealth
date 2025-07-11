#!/bin/bash
# Build script for Vercel deployment with FAISS

echo "Starting build process..."

# Install system dependencies for FAISS
apt-get update -qq
apt-get install -y -qq build-essential cmake

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install Python packages
echo "Installing Python packages..."
pip install -r requirements.txt

echo "Build completed successfully" 