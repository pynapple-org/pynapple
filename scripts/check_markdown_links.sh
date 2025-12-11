#!/bin/bash
set -e

MARKDOWN_LINK_CHECK=$(which markdown-link-check || echo "")
if [[ -z "$MARKDOWN_LINK_CHECK" ]]; then
    echo "❌ ERROR: markdown-link-check command not found. Install it globally via:"
    exit 1
fi

# Check if a link type was specified
LINK_TYPE=$1
if [[ -z "$LINK_TYPE" ]]; then
    echo "❌ Usage: $0 [relative|external|all]"
    exit 1
fi

# Set config path based on type
if [[ "$LINK_TYPE" == "relative" ]]; then
    CONFIG=".mlc.relative.json"
    LABEL="relative"
elif [[ "$LINK_TYPE" == "external" ]]; then
    CONFIG=".mlc.external.json"
    LABEL="external"
elif [[ "$LINK_TYPE" == "all" ]]; then
    CONFIGS=(".mlc.relative.json" ".mlc.external.json")
    LABEL="all"
else
    echo "❌ Unknown link type: $LINK_TYPE"
    echo "Valid options are: relative, external, all"
    exit 1
fi

echo "🔍 Checking $LABEL Markdown links..."
LOG_FILE=$(mktemp)

run_check() {
    local CONFIG=$1
    echo "🔎 Using config: $CONFIG"

    # Check root directory
    echo "📁 Checking root directory..."
    for file in $(find . -maxdepth 1 -name "*.md"); do
        echo "📄 Checking $file..."
        $MARKDOWN_LINK_CHECK -c "$CONFIG" "$file" 2>&1 | tee -a "$LOG_FILE"
    done

    # Check docs directory (up to 2 levels deep) if it exists
    if [[ -d "doc" ]]; then
        echo "📁 Checking docs directory..."
        for file in $(find doc -maxdepth 2 -name "*.md"); do
            echo "📄 Checking $file..."
            $MARKDOWN_LINK_CHECK -c "$CONFIG" "$file" 2>&1 | tee -a "$LOG_FILE"
        done
    fi
}

# Run one or both checks
if [[ "$LINK_TYPE" == "all" ]]; then
    for config in "${CONFIGS[@]}"; do
        run_check "$config"
    done
else
    run_check "$CONFIG"
fi

# Check for errors
if grep -q "ERROR:" "$LOG_FILE"; then
    echo "🚨 Link check failed! Please fix broken links."
    exit 1
else
    echo "✅ All links passed validation."
fi
