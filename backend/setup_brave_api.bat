@echo off
echo ========================================
echo Brave Search API Setup Script
echo ========================================
echo.

echo This script helps you set up your Brave Search API key.
echo.

echo Step 1: Get your API key
echo -------------------------
echo 1. Go to https://api.search.brave.com/register
echo 2. Create an account (credit card required even for free plans)
echo 3. Copy your API key from the dashboard
echo.

set /p API_KEY="Enter your Brave Search API key: "

if "%API_KEY%"=="" (
    echo Error: No API key provided
    pause
    exit /b 1
)

echo.
echo Step 2: Setting environment variable
echo ------------------------------------
setx BRAVE_API_KEY "%API_KEY%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Environment variable set successfully!
    echo.
    echo Step 3: Testing configuration
    echo ------------------------------
    echo.
    echo Note: You may need to restart your terminal/IDE for the environment variable to take effect.
    echo.
    echo To test your configuration, run:
    echo   python test_brave_api.py
    echo.
) else (
    echo.
    echo ❌ Failed to set environment variable
    echo.
    echo Manual setup:
    echo 1. Open System Properties
    echo 2. Click "Environment Variables"
    echo 3. Add BRAVE_API_KEY with your API key
    echo.
)

pause 