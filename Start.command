#!/bin/bash
cd "$(dirname "$0")" || exit 1
for candidate in ".venv/bin/python" "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12" "python3"; do
    if command -v "$candidate" >/dev/null 2>&1; then
        "$candidate" START_HERE.py
        exit $?
    fi
done
echo "Install Python 3.12 from python.org, then double-click this file again."
read -r -p "Press Return to close."
