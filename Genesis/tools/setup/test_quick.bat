@echo off
echo Testing Python environment...
.venv\Scripts\python.exe --version
echo.
echo Testing Blender...
"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" --version
echo.
echo Testing environment script...
.venv\Scripts\python.exe test_environment.py
pause
