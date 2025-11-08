@echo off 
echo Starting Fixfaceapi (React Development Server)... 
echo. 
 
REM Change to the fixfaceapi directory 
cd /d C:\\Users\\b_had\\Documents\\Fixfaceapi 
 
REM Check if node_modules exists 
if not exist node_modules ( 
    echo Installing dependencies first... 
    call npm install 
    echo. 
) 
 
REM Start the development server 
echo Starting development server... 
echo Press Ctrl+C to stop the server 
echo. 
start "" http://localhost:5173 
npm run dev 
 
pause
