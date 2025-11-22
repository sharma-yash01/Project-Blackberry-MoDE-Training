#!/bin/bash
#
# remove_tracked_cache.sh - Remove cache files and ignored files from git tracking
#
# This script removes files that are already tracked by git but should be ignored
# (like Python cache files, .md files except README.md, etc.).
#
# It does NOT delete the files locally - it only removes them from git's tracking.
# After running this, you should commit and push to update the remote repository.
#
# Usage:
#   ./scripts/remove_tracked_cache.sh [--dry-run]
#
# Example:
#   # See what would be removed (dry run)
#   ./scripts/remove_tracked_cache.sh --dry-run
#
#   # Actually remove the files from git tracking
#   ./scripts/remove_tracked_cache.sh

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

DRY_RUN=false

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to mode-training-package directory
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"

# Change to package directory
cd "$PACKAGE_DIR" || exit 1

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a git repository!"
    exit 1
fi

print_info "Working directory: $PACKAGE_DIR"
print_info "Searching for files that should be removed from git tracking..."
echo ""

# Find files that are tracked but should be ignored
# We'll use git ls-files to find tracked files, then check if they match ignore patterns

# Pattern 1: Python cache files (__pycache__ directories and .pyc files)
CACHE_FILES=$(git ls-files | grep -E "(__pycache__|\.pyc)" || true)

# Pattern 2: .md files except README.md
MD_FILES=$(git ls-files | grep "\.md$" | grep -v "^README\.md$" || true)

# Pattern 3: .zip files
ZIP_FILES=$(git ls-files | grep "\.zip$" || true)

# Combine all files to remove
ALL_FILES_TO_REMOVE=$(echo -e "$CACHE_FILES\n$MD_FILES\n$ZIP_FILES" | grep -v "^$" | sort -u)

if [ -z "$ALL_FILES_TO_REMOVE" ]; then
    print_success "No cache files or ignored files are currently tracked by git!"
    print_info "Your repository is clean. The .gitignore will prevent these files from being added in the future."
    exit 0
fi

print_warning "Found the following tracked files that should be ignored:"
echo "----------------------------------------"
echo "$ALL_FILES_TO_REMOVE"
echo "----------------------------------------"
echo ""

# Count files
FILE_COUNT=$(echo "$ALL_FILES_TO_REMOVE" | wc -l | tr -d ' ')
print_info "Total files to remove from git tracking: $FILE_COUNT"

if [ "$DRY_RUN" = true ]; then
    echo ""
    print_warning "DRY RUN MODE - No changes will be made"
    print_info "To actually remove these files, run: ./scripts/remove_tracked_cache.sh"
    exit 0
fi

echo ""
print_warning "These files will be removed from git tracking (but kept locally)"
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Cancelled."
    exit 0
fi

echo ""
print_info "Removing files from git tracking..."

# Remove files from git's index (staging area) but keep them locally
# Using --cached flag removes from index but not from filesystem
# Using -r for recursive removal of directories
REMOVED_COUNT=0
while IFS= read -r file; do
    if [ -n "$file" ]; then
        print_info "  Removing: $file"
        if git rm --cached -r "$file" 2>/dev/null; then
            REMOVED_COUNT=$((REMOVED_COUNT + 1))
        else
            print_warning "  Could not remove: $file (may already be removed)"
        fi
    fi
done <<< "$ALL_FILES_TO_REMOVE"

echo ""
if [ $REMOVED_COUNT -gt 0 ]; then
    print_success "Removed $REMOVED_COUNT file(s) from git tracking!"
    echo ""
    print_info "Next steps:"
    print_info "1. Commit these changes: git commit -m 'Remove cache files and ignored files from tracking'"
    print_info "2. Push to remote: git push"
    print_info ""
    print_info "Or use the run_and_commit.sh script:"
    print_info "  ./scripts/run_and_commit.sh --message 'Remove cache files from git tracking'"
else
    print_warning "No files were removed. They may have already been removed."
fi

