#!/bin/bash
set -e

cd "$(dirname "$0")"

TEX_FILE="cudaHeat.tex"

PYTHON_TASKS=(
    "pythonScripts/latex_event_figure.py 7b6805fd-6c65-406b-8916-7237c2a0b469[0:1,0:2,0:3,0:4,0:5] tikz/events"
    "pythonScripts/generate_rules_catalogue.py ../rules tables"
)

# Calculate column widths and task count
TOTAL=${#PYTHON_TASKS[@]}

SCRIPT_WIDTH=${#SCRIPT}
INPUT_WIDTH=${#INPUT}
OUTPUT_WIDTH=${#OUTPUT}

for task in "${PYTHON_TASKS[@]}"; do
    read -r SCRIPT INPUT OUTPUT <<< "$task"

    DISPLAY_SCRIPT="${SCRIPT#pythonScripts/}"

    (( ${#DISPLAY_SCRIPT} > SCRIPT_WIDTH )) && SCRIPT_WIDTH=${#DISPLAY_SCRIPT}
    (( ${#INPUT} > INPUT_WIDTH )) && INPUT_WIDTH=${#INPUT}
    (( ${#OUTPUT} > OUTPUT_WIDTH )) && OUTPUT_WIDTH=${#OUTPUT}
done

SCRIPT_WIDTH=$((SCRIPT_WIDTH + 2))
INPUT_WIDTH=$((INPUT_WIDTH + 2))
OUTPUT_WIDTH=$((OUTPUT_WIDTH + 2))

FORMAT="%-${SCRIPT_WIDTH}s %-${INPUT_WIDTH}s %-${OUTPUT_WIDTH}s\n"

# Print task summary
echo "Python scripts to execute: $TOTAL"
echo ""

printf "%-8s $FORMAT" "TASK" "SCRIPT" "INPUT" "OUTPUT"

LINE_WIDTH=$((8 + SCRIPT_WIDTH + INPUT_WIDTH + OUTPUT_WIDTH + 2))
printf '%*s\n' "$LINE_WIDTH" '' | tr ' ' '='

# Execute Python scripts
COUNT=0

for task in "${PYTHON_TASKS[@]}"; do
    read -r SCRIPT INPUT OUTPUT <<< "$task"

    COUNT=$((COUNT + 1))
    DISPLAY_SCRIPT="${SCRIPT#pythonScripts/}"

    printf "%-8s $FORMAT" \
        "[$COUNT/$TOTAL]" \
        "$DISPLAY_SCRIPT" \
        "$INPUT" \
        "$OUTPUT"

    python "$SCRIPT" "$INPUT" "$OUTPUT"
done

echo ""
echo "Python scripts completed: $COUNT/$TOTAL"
echo ""

# Compile LaTeX
echo "Starting 1st compilation"
pdflatex -shell-escape -interaction=nonstopmode -halt-on-error "$TEX_FILE" > /dev/null

echo "Starting 2nd compilation"
# pdflatex -shell-escape -interaction=nonstopmode -halt-on-error "$TEX_FILE" > /dev/null

echo "Compilation completed"
echo ""