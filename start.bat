@echo off 
@echo off
echo.
echo Starting Fixfaceapi (React Development Server)...
echo.
cd /d "D:\Projects\lotte\Fixfaceapi"

REM Add Node.js to PATH
set PATH=%PATH%;"C:\Program Files\nodejs"

REM Check if node_modules exists
if not exist node_modules (
    echo Installing dependencies first...
    call npm install
    echo.
)

REM Start the development server
echo Starting development server...
echo Press Ctrl+C in this window to stop the client server
echo.
start "" chrome "http://localhost:5173"
start /min "Fixfaceapi" cmd /k npm run dev

pause
