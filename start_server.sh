#!/bin/bash
source venv/bin/activate
echo "Starting server..."
uvicorn server:app --reload
