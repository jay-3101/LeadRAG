#!/bin/bash

# Exit if any command fails
set -e

# Step 1: Set environment variable (modify as needed)
export API_KEY="your_actual_api_key_here"

# Step 2: Install dependencies
pip install --no-cache-dir -r requirements.txt

# Step 3: Run the app from the backend directory
python backend/app.py
