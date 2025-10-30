#!/bin/sh
# Minimal entrypoint: ensure PORT is set, print startup info, and exec the provided command
set -e

# Default port
: ${PORT:=8502}

echo "Starting container: binding to 0.0.0.0:${PORT}"
echo "Current user: $(id -u):$(id -g)" 2>/dev/null || true

# If no arguments were passed, run gunicorn with sensible defaults
if [ "$#" -eq 0 ]; then
  exec gunicorn --bind "0.0.0.0:${PORT}" drug_resistant_tuberculosis.application:app --workers 1 --threads 4
else
  # If arguments provided, substitute ${PORT} if present and exec
  # Use 'sh -c' to allow shell expansion in the provided command string
  exec sh -c "$*"
fi
