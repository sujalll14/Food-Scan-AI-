#!/bin/bash
# Setup script for macOS/Linux

echo "Setting up FoodLens Environment..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "python3 could not be found. Please install Python 3."
    exit
fi

# Create virtual environment
echo "Creating virtual environment 'venv'..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! To activate the environment, run: source venv/bin/activate"
echo "Then, you can train the model with: python train.py"
echo "And start the web app with: python app.py"
