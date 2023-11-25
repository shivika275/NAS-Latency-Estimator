#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <folder_name> <results_folder> <timing_csv_file> <simpleperf_csv_file>"
    exit 1
fi

FOLDER_NAME=$1
RESULTS_FOLDER=$2
TIMING_CSV_FILE=$3
SIMPLEPERF_CSV_FILE=$4

# Create the results folder if it doesn't exist
mkdir -p "$RESULTS_FOLDER"

# Create CSV files with headers if they don't exist
if [ ! -f "$TIMING_CSV_FILE" ]; then
    echo "Model Name,Initialization Time (ms),First Inference Time (us),Warmup Average Time (us),Inference Average Time (us),Memory Footprint Init (MB),Memory Footprint Overall (MB)" > "$TIMING_CSV_FILE"
fi

if [ ! -f "$SIMPLEPERF_CSV_FILE" ]; then
    echo "modelname,cpu-cycles-samples,cpu-cycles-count,cache-references-samples,cache-references-count,cache-misses-samples,cache-misses-count,instructions-samples,instructions-count" > "$SIMPLEPERF_CSV_FILE"
fi

# Path to the parsing scripts
INF_PARSE_SCRIPT_PATH="./process_inf_info.py"
SIMPLEPERF_PARSE_SCRIPT_PATH="./process_simpleperf_info.py"

# Traverse the folder for all .tflite files
for FILE in $FOLDER_NAME/*.tflite; do
    if [ -f "$FILE" ]; then
        # Extract the base name of the file
        BASENAME=$(basename "$FILE")
        MODEL_NAME=$(echo "$BASENAME" | sed 's/\.tflite$//') # Remove .tflite extension

        # Push the file to the device
        adb push "$FILE" /data/local/tmp/$BASENAME

        # Run benchmark 50 times
        adb shell taskset f0 /data/local/tmp/android_aarch64_benchmark_model --graph=/data/local/tmp/$BASENAME --enable_op_profiling=true > "$RESULTS_FOLDER/res-$MODEL_NAME.txt"

        # Start simpleperf record with specified events and run the benchmark
        adb shell simpleperf record -e cpu-cycles,cache-references,cache-misses,instructions -o /data/local/tmp/simpleperf_data \
        taskset f0 /data/local/tmp/android_aarch64_benchmark_model --graph=/data/local/tmp/$BASENAME --enable_op_profiling=true --num_runs=1 >"simpleperf_inf_$MODEL_NAME.txt"

        # Pull the simpleperf data file
        adb pull /data/local/tmp/simpleperf_data "simpleperf_report-$MODEL_NAME.data"

        # Generate the simpleperf report
        adb shell simpleperf report -i /data/local/tmp/simpleperf_data > "cache_report-$MODEL_NAME.txt"

        SIMPLEPERF_REPORT_FILE="cache_report-$MODEL_NAME.txt"

        python3 "$SIMPLEPERF_PARSE_SCRIPT_PATH" "$SIMPLEPERF_REPORT_FILE" "$SIMPLEPERF_CSV_FILE"

        # Call the parsing script for inference and memory data
        python3 "$INF_PARSE_SCRIPT_PATH" "$RESULTS_FOLDER/res-$MODEL_NAME.txt" "$TIMING_CSV_FILE"

        # Cleanup: remove temporary files
        rm -f "simpleperf_report-$MODEL_NAME.data" "cache_report-$MODEL_NAME.txt" "simpleperf_inf_$MODEL_NAME.txt"
        rm -f "$RESULTS_FOLDER/res-$MODEL_NAME.txt"
        adb shell rm -f /data/local/tmp/simpleperf_data /data/local/tmp/$BASENAME
    fi
done

echo "Timing CSV file generated: $TIMING_CSV_FILE"
echo "Simpleperf CSV file generated: $SIMPLEPERF_CSV_FILE"
