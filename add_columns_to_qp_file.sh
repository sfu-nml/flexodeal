#!/bin/bash

# Usage: ./add_columns_inplace.sh file.csv col1 val1 col2 val2 ...

FILE=$1
TEMP_FILE=$(mktemp)

# Check if there are at least three arguments and the number of column-value pairs is correct
if [[ $# -lt 3 || $(( ($# - 1) % 2 )) -ne 0 ]]; then
  echo "Usage: $0 file.csv col1 val1 col2 val2 ..."
  exit 1
fi

# Shift to process column-value pairs
shift

COLUMN_NAMES=()
CONSTANT_VALUES=()

while [[ $# -gt 0 ]]; do
  COLUMN_NAMES+=("$1")  # Add column name
  CONSTANT_VALUES+=("$2")  # Add constant value
  shift 2  # Move to the next column-value pair
done

# Prepare the additional headers and values for appending
ADD_HEADERS=$(IFS=,; echo "${COLUMN_NAMES[*]}")
ADD_VALUES=$(IFS=,; echo "${CONSTANT_VALUES[*]}")

# Add the new columns to the CSV
awk -v add_headers="$ADD_HEADERS" -v add_values="$ADD_VALUES" '
BEGIN { FS=OFS="," }
NR == 1 { print $0, add_headers }  # Add new headers to the first row
NR > 1 { print $0, add_values }    # Add constant values to other rows
' "$FILE" > "$TEMP_FILE"

# Overwrite the original file with the updated content
mv "$TEMP_FILE" "$FILE"

echo "New columns added successfully to $FILE"
