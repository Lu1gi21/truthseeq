# Brave Search API Setup Script for PowerShell

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Brave Search API Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "This script helps you set up your Brave Search API key." -ForegroundColor Yellow
Write-Host ""

Write-Host "Step 1: Get your API key" -ForegroundColor Green
Write-Host "-------------------------" -ForegroundColor Green
Write-Host "1. Go to https://api.search.brave.com/register" -ForegroundColor White
Write-Host "2. Create an account (credit card required even for free plans)" -ForegroundColor White
Write-Host "3. Copy your API key from the dashboard" -ForegroundColor White
Write-Host ""

$apiKey = Read-Host "Enter your Brave Search API key" -AsSecureString
$apiKeyPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($apiKey))

if ([string]::IsNullOrEmpty($apiKeyPlain)) {
    Write-Host "Error: No API key provided" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Step 2: Setting environment variable" -ForegroundColor Green
Write-Host "------------------------------------" -ForegroundColor Green

try {
    [Environment]::SetEnvironmentVariable("BRAVE_API_KEY", $apiKeyPlain, "User")
    Write-Host ""
    Write-Host "✅ Environment variable set successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Step 3: Testing configuration" -ForegroundColor Green
    Write-Host "-----------------------------" -ForegroundColor Green
    Write-Host ""
    Write-Host "Note: You may need to restart your terminal/IDE for the environment variable to take effect." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To test your configuration, run:" -ForegroundColor White
    Write-Host "  python test_brave_api.py" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host ""
    Write-Host "❌ Failed to set environment variable" -ForegroundColor Red
    Write-Host ""
    Write-Host "Manual setup:" -ForegroundColor Yellow
    Write-Host "1. Open System Properties" -ForegroundColor White
    Write-Host "2. Click 'Environment Variables'" -ForegroundColor White
    Write-Host "3. Add BRAVE_API_KEY with your API key" -ForegroundColor White
    Write-Host ""
}

Read-Host "Press Enter to exit" 