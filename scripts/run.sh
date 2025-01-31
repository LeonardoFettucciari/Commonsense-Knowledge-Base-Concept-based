#!/bin/bash

# Check if the first argument (API key) is missing
if [ -z "$1" ]; then
    echo "Usage: $0 <api_key>"
    exit 1
fi

# Store the API key
API_KEY="$1"

python src/run.py --api_key "$API_KEY"