#!/bin/bash

# Add the workspace to the safe directory
git config --global --add safe.directory /workspaces/BlueMath
# Get the current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $BRANCH"

if [ "$BRANCH" != "main" ]; then
    echo "Installing bluemath-tk and pymesh2d in development mode..."
    # Clone the repositories
    git clone https://github.com/GeoOcean/BlueMath_tk.git /workspaces/BlueMath_tk
    git clone https://github.com/GeoOcean/pymesh2d.git /workspaces/pymesh2d
    # Install in development mode
    cd /workspaces/BlueMath_tk
    pip install .  # -e flag for development mode
    cd /workspaces/pymesh2d
    pip install .  # -e flag for development mode
else
    echo "Not in develop branch, skipping development installation"
fi