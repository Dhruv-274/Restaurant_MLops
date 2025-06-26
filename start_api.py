#!/bin/bash
# start_api.py
import uvicorn
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

if __name__ == "__main__":
    # Start the API server
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info"
    )
