#!/usr/bin/env bash
# Download the IMC'25 labeled smishing dataset (CC BY 4.0) at a pinned commit.
set -euo pipefail
COMMIT=a6175560b57387199871e51fbef6bc523d2516b4
URL="https://raw.githubusercontent.com/reportsmishing/Smishing-Dataset-IMC25/${COMMIT}/dataset/final_dataset_output.csv"
mkdir -p data
curl -fsSL "$URL" -o data/final_dataset_output.csv
echo "Saved data/final_dataset_output.csv ($(wc -l < data/final_dataset_output.csv) lines)"
