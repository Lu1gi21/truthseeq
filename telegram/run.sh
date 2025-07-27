#!/bin/bash

echo "Starting Social Media Downloader Telegram Bot..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if requirements are installed
echo "Checking dependencies..."
if ! python3 -c "import telegram" &> /dev/null; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
fi

# Create downloads directory if it doesn't exist
mkdir -p downloads

# Run the bot
echo "Starting bot..."
python3 bot.py 