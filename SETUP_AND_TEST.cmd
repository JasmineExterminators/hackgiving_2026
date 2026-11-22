@echo off
setlocal
cd /d "%~dp0"
echo Turkey Keyboard - setup and webcam preview
echo This creates a private Python environment in this project folder.
echo It downloads the required libraries and a pose model.
echo.
if exist ".venv\Scripts\python.exe" goto packages
py -3.13 -m venv .venv
if errorlevel 1 goto failed
:packages
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "import cv2, serial, pyautogui; from mediapipe.tasks.python import vision; print('Required libraries imported successfully.')"
if errorlevel 1 goto failed
echo.
echo Opening webcam simulation. No Arduino is needed.
echo Positions: diagonal down = first, T = second, straight up = third.
echo Click the preview and use 1-9 to simulate the nine foot pads.
echo O switches modes. Y accepts a suggestion. X deletes a character.
echo Letters: straight UP plus pad 9 = Space.
echo Press Q in the preview to quit. Typing into other apps is disabled.
".venv\Scripts\python.exe" turkey_keyboard.py --simulate --pads 9
if errorlevel 1 goto failed
goto done
:failed
echo.
echo Setup or testing stopped. Please share the error text above in the chat.
echo If pip reports no compatible MediaPipe version, Python 3.12 may be needed.
:done
pause
