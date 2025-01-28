#!/bin/bash

input_dir="/data/MyDataBase/images"
output_dir="/data/MyDataBase/images"

# Desired resolution
target_width=1024
target_height=512

for file in "$input_dir"/*_0.mp4; do
  if [[ -f "$file" ]]; then
    output_file="${output_dir}/$(basename "$file" .mp4)_fixed.mp4"
    echo "Re-encoding $file to $output_file with resolution ${target_width}x${target_height}"
    ffmpeg -i "$file" -vf "scale=${target_width}:${target_height}" -vcodec libx264 -crf 23 -preset fast -acodec aac -strict -2 "$output_file"
  fi
done
