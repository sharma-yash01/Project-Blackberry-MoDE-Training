#!/bin/bash
#
# run_and_commit.sh - Quickly commit and push changes to GitHub
#
# This script automates the workflow of:
# 1. Staging all your current changes
# 2. Creating a git commit with a descriptive message (auto-generated or custom)
# 3. Pushing to the remote repository
#
# This is designed for quick commits when you've made changes and just want
# to commit and push them without running any training scripts.
#
# Usage:
#   ./scripts/run_and_commit.sh [OPTIONS]
#
# Examples:
#   # Quick commit and push with auto-generated message
#   ./scripts/run_and_commit.sh
#
#   # Commit and push with custom message
#   ./scripts/run_and_commit.sh --message "Fixed bug in trainer.py"
#
#   # Just commit without pushing (use --no-push)
#   ./scripts/run_and_commit.sh --no-push
#
#   # See what would be committed without actually doing it (dry run)
#   ./scripts/run_and_commit.sh --dry-run

set -e  # Exit on any error - this ensures we don't commit if something fails

# Color codes for output (makes it easier to read)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values for options
COMMIT_MSG=""  # Will be auto-generated if not provided
PUSH=true  # Default to pushing after commit (can be disabled with --no-push)
DRY_RUN=false  # If true, show what would be done without actually doing it

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

# Function to display usage information
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -m, --message MSG      Custom commit message (default: auto-generated)
    --no-push              Commit but don't push to remote
    --dry-run              Show what would be done without actually doing it
    -h, --help             Show this help message

Examples:
    # Quick commit and push with auto-generated message
    $0

    # Commit and push with custom message
    $0 --message "Update README with new configuration"

    # Just commit without pushing
    $0 --no-push

    # See what would happen (dry run)
    $0 --dry-run
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--message)
            COMMIT_MSG="$2"
            shift 2
            ;;
        --no-push)
            PUSH=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
done

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to mode-training-package directory (parent of scripts)
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"

# Change to package directory to ensure we're in the right place
cd "$PACKAGE_DIR" || exit 1

print_info "Working directory: $PACKAGE_DIR"
print_info "Current git branch: $(git branch --show-current)"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a git repository! Please run this from the mode-training-package directory."
    exit 1
fi

# If dry run, just show what would happen
if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN MODE - No changes will be made"
    echo ""
    
    # Show what files would be staged
    GIT_STATUS=$(git status --short)
    if [ -z "$GIT_STATUS" ]; then
        echo "No changes detected. Nothing would be committed."
    else
        echo "Files that would be staged:"
        echo "----------------------------------------"
        git status --short
        echo "----------------------------------------"
        echo ""
        
        # Show what commit message would be used
        if [ -z "$COMMIT_MSG" ]; then
            echo "Would auto-generate commit message based on changes"
        else
            echo "Would commit with message: $COMMIT_MSG"
        fi
        
        if [ "$PUSH" = true ]; then
            echo "Would push to remote after committing"
        else
            echo "Would NOT push to remote"
        fi
    fi
    exit 0
fi

# Step 1: Check git status to see what changed
print_info "Checking git status..."
GIT_STATUS=$(git status --short)

if [ -z "$GIT_STATUS" ]; then
    print_warning "No changes detected. Nothing to commit."
    exit 0
fi

echo ""
print_info "Changes detected:"
echo "----------------------------------------"
git status --short
echo "----------------------------------------"
echo ""

# Step 2: Stage all changes
# This stages all modified, new, and deleted files
print_info "Staging all changes..."
git add -A

# Verify something was staged
if git diff --cached --quiet; then
    print_warning "No changes to commit after staging."
    exit 0
fi

print_success "Changes staged successfully!"

# Step 3: Generate commit message if not provided
if [ -z "$COMMIT_MSG" ]; then
    # Auto-generate commit message based on what was changed
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    
    # Count modified and new files
    MODIFIED=$(git diff --cached --name-only | wc -l | tr -d ' ')
    
    # Try to get a summary of key changes by looking at which files changed
    CHANGED_FILES=$(git diff --cached --name-only)
    
    # Check for different types of changes to create a more descriptive message
    if echo "$CHANGED_FILES" | grep -q "train_mode.py\|trainer.py\|train_supervised.py"; then
        COMMIT_MSG="Update training scripts - $MODIFIED files changed ($TIMESTAMP)"
    elif echo "$CHANGED_FILES" | grep -q "config.py"; then
        COMMIT_MSG="Update configuration - $MODIFIED files changed ($TIMESTAMP)"
    elif echo "$CHANGED_FILES" | grep -q "README.md"; then
        COMMIT_MSG="Update documentation - $MODIFIED files changed ($TIMESTAMP)"
    elif echo "$CHANGED_FILES" | grep -q "model.py"; then
        COMMIT_MSG="Update model architecture - $MODIFIED files changed ($TIMESTAMP)"
    elif echo "$CHANGED_FILES" | grep -q "objectives.py"; then
        COMMIT_MSG="Update loss functions/objectives - $MODIFIED files changed ($TIMESTAMP)"
    elif echo "$CHANGED_FILES" | grep -q "scripts/"; then
        COMMIT_MSG="Update scripts - $MODIFIED files changed ($TIMESTAMP)"
    else
        COMMIT_MSG="Update package - $MODIFIED files changed ($TIMESTAMP)"
    fi
    
    print_info "Auto-generated commit message: $COMMIT_MSG"
else
    print_info "Using provided commit message: $COMMIT_MSG"
fi

# Step 4: Create the commit
print_info "Creating commit..."
git commit -m "$COMMIT_MSG"

COMMIT_HASH=$(git rev-parse --short HEAD)
print_success "Commit created successfully!"
print_info "Commit hash: $COMMIT_HASH"
echo ""

# Step 5: Push to remote (if enabled)
if [ "$PUSH" = true ]; then
    print_info "Pushing to remote repository..."
    
    # Get the current branch name
    CURRENT_BRANCH=$(git branch --show-current)
    
    # Check if remote exists
    if ! git remote | grep -q .; then
        print_error "No remote repository configured!"
        print_error "To add a remote: git remote add origin <url>"
        exit 1
    fi
    
    # Get remote name (usually 'origin')
    REMOTE_NAME=$(git remote | head -n 1)
    
    # Push to remote
    if git push "$REMOTE_NAME" "$CURRENT_BRANCH"; then
        print_success "Pushed to remote successfully!"
    else
        print_error "Failed to push to remote!"
        print_error "You may need to pull first or check your permissions."
        exit 1
    fi
else
    print_info "Skipping push (use --no-push to explicitly disable, or omit for auto-push)"
fi

echo ""
print_success "All done! ✓"
print_info "Summary:"
echo "  - Files changed: $(git diff --cached --name-only 2>/dev/null | wc -l | tr -d ' ' || echo "0")"
echo "  - Commit hash: $COMMIT_HASH"
echo "  - Commit message: $COMMIT_MSG"
echo "  - Pushed to remote: $([ "$PUSH" = true ] && echo "Yes" || echo "No")"
