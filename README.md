# FoodLens

FoodLens is an AI-powered Full-Stack Web Application that uses a Convolutional Neural Network (CNN) trained with Transfer Learning (MobileNetV2) to identify food items from images. It gives you the nutritional breakdown (Calories, Protein, Carbs) of the identified food.

It supports:
- Uploading up to 10 images simultaneously.
- Taking photos directly from the device's camera.
- A beautiful, glassmorphic UI.
- Very fast inference.

## Project Structure
```text
/foodlens/
├── train.py          ← Downloads 10-class subset of Food-101 + trains MobileNetV2
├── predict.py        ← CLI inference test tool
├── app.py            ← Flask server and /predict API endpoint
├── nutrition.json    ← Nutritional mapping for the 10 classes
├── requirements.txt  ← All Python dependencies
├── setup.sh          ← One-click setup for macOS/Linux
├── setup.bat         ← One-click setup for Windows
└── static/
    └── index.html    ← The beautiful FoodLens UI via Vanilla JS+CSS
```

## Quick Start (Setup)

### Option 1: Using the provided scripts (Recommended)
**Windows:**
Double click `setup.bat` or run:
```bat
setup.bat
```

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup
**1. Setup environment**
```bash
python -m venv venv
```
Activate it:
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage Guide

### Step 1: Train the Model (Required first step)
Before you can run the web app or predict via CLI, you must train the model. This script automatically downloads a lightweight subset of the Food-101 dataset (~220MB) and trains a transfer-learning model.
```bash
python train.py --epochs 5
```
This produces `foodlens_model.h5` and `class_indices.json`.

### Step 2: Use the CLI Predictor (Optional)
Test the trained model locally without starting a server:
```bash
python predict.py "path/to/some/image.jpg"
```

### Step 3: Run the Web App
Start the Flask server:
```bash
python app.py
```
Then, open your browser and go to `http://localhost:5000`

## Available Food Classes
The model trained by `train.py` is capable of identifying 10 items.
1. Chicken Curry
2. Chicken Wings
3. Fried Rice
4. Grilled Salmon
5. Hamburger
6. Ice Cream
7. Pizza
8. Ramen
9. Steak
10. Sushi

Enjoy analyzing your food!
