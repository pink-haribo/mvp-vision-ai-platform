@echo off
echo Running labelme2coco_simple.py in iam-py310 environment...
call "C:\Users\brightyoun\miniconda3\Scripts\activate.bat" iam-py310
if errorlevel 1 (
    echo Failed to activate conda environment
    exit /b 1
)
echo Environment activated successfully
echo.
echo Running labelme2coco conversion...
python labelme2coco_simple.py train train_annotations --label labels.txt --noviz
