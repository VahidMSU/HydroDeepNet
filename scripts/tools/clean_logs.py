import os 

path = "/data/SWATGenXApp/codes/web_application/logs"

log_files = os.listdir(path)    

# Maximum file size in bytes (20MB)
MAX_SIZE = 5 * 1024 * 1024

def trim_log_file(file_path, max_size):
    """Trim log file to be under the maximum size by removing old lines."""
    # Check current file size
    file_size = os.path.getsize(file_path)
    
    if file_size <= max_size:
        print(f"File {file_path} is already under {max_size/1024/1024:.2f}MB, skipping.")
        return
    
    # Optional: Create backup
    # backup_path = file_path + ".backup"
    # with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
    #     dst.write(src.read())
    
    # Read all lines
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Calculate approximately how many lines we need to remove
    total_lines = len(lines)
    approx_line_size = file_size / total_lines
    lines_to_keep = int(max_size / approx_line_size)
    
    # Keep the most recent lines (at the end of the file)
    lines_to_keep = min(lines_to_keep, total_lines)
    trimmed_lines = lines[-lines_to_keep:]
    
    # Write trimmed content back to file
    with open(file_path, 'w') as f:
        f.writelines(trimmed_lines)
    
    new_size = os.path.getsize(file_path)
    print(f"Trimmed {file_path} from {file_size/1024/1024:.2f}MB to {new_size/1024/1024:.2f}MB")

# Process each log file
for log_file in log_files:
    file_path = os.path.join(path, log_file)
    
    # Skip directories and non-regular files
    if not os.path.isfile(file_path):
        continue
    
    try:
        trim_log_file(file_path, MAX_SIZE)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print("Log cleaning completed.")