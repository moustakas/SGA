#!/bin/bash
file="$1"
result=$(grep -ao "VER_FITB.\{0,30\}" "$file" | cut -d"'" -f2)
if [[ -n "$result" ]]; then
    group=$(basename $(dirname "$file"))
    echo "$group,$result"
fi
