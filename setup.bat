@echo off
REM Setup script for Windows

echo Setting up FoodLens Environment...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python could not be found. Please install Python 3 and add it to PATH.
    pause
    exit /b
)

REM Create virtual environment
echo Creating virtual environment 'venv'...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Setup complete! 
echo To activate the environment, run: venv\Scripts\activate
echo Then, you can train the model with: python train.py
echo And start the web app with: python app.py
pause
