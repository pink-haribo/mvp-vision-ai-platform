@echo off
echo Checking iam-py310 environment...
call "C:\Users\brightyoun\miniconda3\Scripts\activate.bat" iam-py310
if errorlevel 1 (
    echo Failed to activate conda environment
    exit /b 1
)
echo Environment activated successfully
echo.
echo Python executable:
python -c "import sys; print(sys.executable)"
echo.
echo Checking labelme installation:
python -c "import labelme; print('labelme version:', labelme.__version__)" 2>nul || echo "labelme not found"
echo.
echo Checking pycocotools installation:
python -c "import pycocotools; print('pycocotools found')" 2>nul || echo "pycocotools not found"
echo.
echo Installed packages in environment:
pip list | findstr -i "labelme pycocotools"
