@echo off
echo Installing labelme in iam-py310 environment...
call "C:\Users\brightyoun\miniconda3\Scripts\activate.bat" iam-py310
if errorlevel 1 (
    echo Failed to activate conda environment
    exit /b 1
)
echo Environment activated successfully
echo.
echo Installing labelme...
pip install labelme
echo.
echo Installation complete. Verifying...
python -c "import labelme; print('labelme version:', labelme.__version__)"
