#!/bin/bash
#
# Shell script to create a properly structured zip file of mode-training-package.
#
# This script:
# 1. Navigates to the mode-training-package directory
# 2. Creates a zip file with setup.py at the root level (not nested)
# 3. Places the zip file in the Project-Blackberry-MoDE-Training directory
# 4. Excludes unnecessary files (.DS_Store, __pycache__, etc.)
#

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to mode-training-package directory (parent of scripts)
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"

# Navigate to Project-Blackberry-MoDE-Training directory (parent of mode-training-package)
PROJECT_DIR="$(dirname "$PACKAGE_DIR")"

# Output zip file path
ZIP_FILE="$PROJECT_DIR/mode-training-package.zip"

# Change to package directory to ensure proper zip structure
cd "$PACKAGE_DIR" || exit 1

echo "=========================================="
echo "Creating mode-training-package.zip"
echo "=========================================="
echo "Package directory: $PACKAGE_DIR"
echo "Output zip file: $ZIP_FILE"
echo ""

# Remove old zip if it exists
if [ -f "$ZIP_FILE" ]; then
    echo "Removing existing zip file..."
    rm -f "$ZIP_FILE"
fi

# Create zip file with proper structure (files at root, not nested)
# Exclude unnecessary files
echo "Creating zip file..."
zip -r "$ZIP_FILE" . \
    -x "*.DS_Store" \
    -x "__pycache__/*" \
    -x "*.pyc" \
    -x ".git/*" \
    -x "__MACOSX/*" \
    -x "*.zip" \
    -x ".gitignore" \
    -x "*.swp" \
    -x "*.swo" \
    -x "*~"

# Check if zip was created successfully
if [ -f "$ZIP_FILE" ]; then
    echo ""
    echo "=========================================="
    echo "✓ Zip file created successfully!"
    echo "=========================================="
    echo "Location: $ZIP_FILE"
    echo ""
    
    # Show first few entries to verify structure
    echo "Verifying zip structure (first 10 entries):"
    echo "-------------------------------------------"
    unzip -l "$ZIP_FILE" | head -12
    echo ""
    
    # Check if setup.py is at root level
    if unzip -l "$ZIP_FILE" | grep -q "^[[:space:]]*[0-9]*[[:space:]]*.*setup.py$"; then
        echo "✓ setup.py found at root level - structure is correct!"
    else
        echo "⚠ WARNING: setup.py not found at root level. Check zip structure."
    fi
    
    # Get file size
    FILE_SIZE=$(du -h "$ZIP_FILE" | cut -f1)
    echo "File size: $FILE_SIZE"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ ERROR: Failed to create zip file"
    echo "=========================================="
    exit 1
fi

