#!/bin/bash

#REGION="dr11-south"
REGION="dr11-north"
OUTFILE="versions-$REGION.csv"
TOPDIR="/pscratch/sd/i/ioannis/SGA2025-v1.4/$REGION"
NJOBS=32

# Build set of already-processed groups
if [[ -f "$OUTFILE" ]]; then
    DONE=$(mktemp)
    awk -F, 'NR>1 {print $1}' "$OUTFILE" > "$DONE"
    echo "Resuming: $(wc -l < "$DONE") groups already done"
else
    DONE=$(mktemp)
    echo "group,version" > "$OUTFILE"
fi

find "$TOPDIR" . -name '*-tractor.fits' -print0 | \
    xargs -0 -P "$NJOBS" -n1 ./sga-version-one.sh | \
    grep -vFf "$DONE" >> "$OUTFILE"

rm -f "$DONE"

TOTAL=$(wc -l < "$OUTFILE")
echo "Done: $((TOTAL - 1)) groups written to $OUTFILE"
