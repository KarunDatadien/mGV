#!/bin/bash
# This script processes all .nc files under /global_data (including subdirectories).
# For each file, it creates a backup, converts float64 to float32 (via CDO),
# and writes out an uncompressed version (via nccopy).

# Function to process an individual file.
process_file() {
    local infile="$1"
    echo "--------------------------------------"
    echo "Processing file: $infile"

    # Compute backup file name by replacing .nc with .backup
    local backup="${infile%.nc}.backup"
    echo "Creating backup: $backup"
    cp "$infile" "$backup" || { echo "Error: Could not create backup for $infile"; return 1; }

    # Create a temporary file for conversion.
    local tmpfile
    tmpfile=$(mktemp /tmp/converted.XXXXXX.nc) || { echo "Error: Could not create temporary file for $infile"; return 1; }

    # Use CDO to convert the file forcing float32 precision.
    echo "Converting data to float (float32) using CDO..."
    cdo -b F32 copy "$infile" "$tmpfile" || { echo "Error: CDO conversion failed for $infile"; rm -f "$tmpfile"; return 1; }

    # Use nccopy to remove any compression and write the result back to the original file.
    echo "Removing compression with nccopy..."
    nccopy -u "$tmpfile" "$infile" || { echo "Error: nccopy failed for $infile"; rm -f "$tmpfile"; return 1; }

    # Clean up the temporary file.
    rm -f "$tmpfile"
    echo "Finished processing file: $infile"
}

# Export the function if needed by sub-shells.
export -f process_file

# Find all .nc files in /global_data (including subdirectories) and process them.
find ../global_data -type f -name '*.nc' -print0 | while IFS= read -r -d '' file; do
    process_file "$file"
done

echo "--------------------------------------"
echo "All files processed."
