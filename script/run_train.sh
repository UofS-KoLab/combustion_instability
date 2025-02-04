#!/bin/bash

# Default values
DATA_ROOT="${DATA_ROOT:-/home/user/combustion_instability/data/raw/h2}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/user/combustion_instability}"
STABILITY_FILE="${STABILITY_FILE:-/home/user/combustion_instability/data/labels/h2_label.csv}"
WINDOW_SIZE="${WINDOW_SIZE:-100}"
DURATION_SAMPLE_MS="${DURATION_SAMPLE_MS:-12000}"
APPROACH="${APPROACH:-fft}"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-/home/user/anaconda3/envs/hidrogen/bin/python}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-/home/user/combustion_instability/src/train_time_series_model.py}"

# Run the Python script with arguments
"$PYTHON_EXECUTABLE" "$PYTHON_SCRIPT" \
    --data_root "$DATA_ROOT" \
    --project_root "$PROJECT_ROOT" \
    --stability_file "$STABILITY_FILE" \
    --window_size "$WINDOW_SIZE" \
    --approach "$APPROACH" \
    --duration_sample_ms "$DURATION_SAMPLE_MS"
