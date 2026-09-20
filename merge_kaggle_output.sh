#!/usr/bin/env bash
# Merge the 03_kaggle_final output into the repo.
#
# Copies ONLY the files that notebook produces. results/tables/ also holds the
# Phase 3 exploratory artefacts and the Phase 4 baselines, which that session
# never had -- several derive from traffic_matrix.npy, which is not committed,
# so replacing the directory wholesale would cost a 19.4 GiB re-ingest.
#
# Files are located by name anywhere under the given directory, because Kaggle
# downloads individual files flat rather than preserving results/tables/.
set -uo pipefail

KAGGLE_OUT="${1:?usage: merge_kaggle_output.sh <directory holding the downloaded files>}"
REPO="$(cd "$(dirname "$0")" && pwd)"

TABLES=(
  final_metrics_all_test.csv final_metrics_all_stress.csv
  final_metrics_area_5059_test.csv final_metrics_area_5059_stress.csv
  final_metrics_area_5259_test.csv final_metrics_area_5259_stress.csv
  final_metrics_area_5161_test.csv final_metrics_area_5161_stress.csv
  timing_test.csv timing_stress.csv
)
PREDICTIONS=(
  test_area_5059.parquet test_area_5259.parquet test_area_5161.parquet
  stress_area_5059.parquet stress_area_5259.parquet stress_area_5161.parquet
)

mkdir -p "$REPO/results/tables" "$REPO/results/predictions"
missing=()
copied=0

merge_one() {  # name, destination directory
  local found
  found="$(find "$KAGGLE_OUT" -maxdepth 3 -name "$1" -type f -print -quit 2>/dev/null)"
  if [ -z "$found" ]; then
    missing+=("$1")
    return
  fi
  cp "$found" "$2/$1"
  echo "  $(basename "$2")/$1"
  copied=$((copied + 1))
}

for f in "${TABLES[@]}";      do merge_one "$f" "$REPO/results/tables"; done
for f in "${PREDICTIONS[@]}"; do merge_one "$f" "$REPO/results/predictions"; done

echo
echo "copied $copied of $(( ${#TABLES[@]} + ${#PREDICTIONS[@]} )) files"
if [ ${#missing[@]} -gt 0 ]; then
  echo "NOT FOUND under $KAGGLE_OUT:"
  printf '  %s\n' "${missing[@]}"
  echo
  echo "Download these from the committed Kaggle version's Output tab."
  exit 1
fi
