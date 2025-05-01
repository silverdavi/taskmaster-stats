#!/bin/bash

echo "======================================"
echo "Taskmaster UK Data Analysis"
echo "======================================"

# Change to the TaskmasterAnalysis directory
cd "$(dirname "$0")"

# Create output folder if it doesn't exist
mkdir -p output
mkdir -p output/series_deep_dives

# First, run the correlation analysis to generate episode metrics
echo "Step 1: Generating episode metrics..."
python code/corr_analysis_wrapper.py

# Then run the visualization analysis
echo "Step 2: Generating visualizations..."
python code/run_analysis.py

echo "======================================"
echo "Analysis complete!"
echo "Results are in the 'output' directory"
echo "======================================" 