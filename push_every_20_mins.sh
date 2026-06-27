#!/bin/bash
set -e  # stop script on error

END_TIME=$(( $(date +%s) + 3*24*60*60 ))  # 3 days from now
RUN=1

while [ "$(date +%s)" -lt "$END_TIME" ]; do
    echo "Run #$RUN at $(date)"

    # Check if there are changes
    if ! git diff --quiet || ! git diff --cached --quiet; then
        git add .
        git commit -m "Added results at $(date '+%Y-%m-%d %H:%M:%S')"
        git push origin main
    else
        echo "No changes to commit"
    fi

    RUN=$((RUN + 1))

    # Sleep 20 minutes unless we're already past the end time
    if [ "$(date +%s)" -lt "$END_TIME" ]; then
        sleep 1200
    fi
done

echo "Finished after 3 days."