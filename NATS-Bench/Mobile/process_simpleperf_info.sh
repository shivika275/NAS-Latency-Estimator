#!/bin/bash

# Check if the simpleperf report file and csv file are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <simpleperf_report_file> <csv_file>"
    exit 1
fi

SIMPLEPERF_REPORT_FILE=$1
CSV_FILE=$2
MODEL_NAME=$(basename "$SIMPLEPERF_REPORT_FILE" | sed 's/cache_report-//; s/\.txt$//')

# Use awk to parse the simpleperf report and append to CSV
awk -v model="$MODEL_NAME" '
    /Event:/ {event = $2}
    /Samples:/ {samples = $2}
    /Event count:/ {count = $3; print model","event","samples","count}' "$SIMPLEPERF_REPORT_FILE" >> "$CSV_FILE"
