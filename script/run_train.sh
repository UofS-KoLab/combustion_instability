#!/bin/bash

# Default values
DATA_ROOT="${DATA_ROOT:-/data/Stephany1/combustion_project/data/raw/h2}"
PROJECT_ROOT="${PROJECT_ROOT:-/data/Stephany1/combustion_project}"
STABILITY_FILE="${STABILITY_FILE:-/data/Stephany1/combustion_project/data/labels/h2_label.csv}"
WINDOW_SIZE="${WINDOW_SIZE:-100}"
DURATION_SAMPLE_MS="${DURATION_SAMPLE_MS:-12000}"
APPROACH="${APPROACH:-time_series}"

#PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-/home/user/anaconda3/envs/hidrogen/bin/python}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-/data/Stephany1/combustion_project/src/train_time_series_model.py}"

# Run the Python script with arguments
python3 "$PYTHON_SCRIPT" \
    --data_root "$DATA_ROOT" \
    --project_root "$PROJECT_ROOT" \
    --stability_file "$STABILITY_FILE" \
    --window_size "$WINDOW_SIZE" \
    --approach "$APPROACH" \
    --duration_sample_ms "$DURATION_SAMPLE_MS"
