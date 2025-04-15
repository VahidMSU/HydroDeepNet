import base64
import os
import ollama
import json
import pandas as pd

# ==== IMAGE ANALYSIS TOOL ====
def describe_image(path, DEFAULT_MODEL, logger, memory=None, prompt="Analyze the image and describe it in detail"):
    logger.info(f"Describing image... {path}")
    
    # Check if we already have this analysis in memory
    if memory:
        # Get file from memory system if it exists
        file_records = memory.get_related_files(f"image {os.path.basename(path)}", ["image"])
        for file in file_records:
            if file.get("original_path") == path:
                logger.info(f"Using cached image description for {path}")
                return file.get("content", '')
    
    with open(path, "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode("utf-8")
    response = ollama.chat(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "user", "content": prompt, "images": [img_data]}
        ]
    )
    result = response["message"]["content"]
    
    # Store in memory if available
    if memory:
        memory.add_file(path, result, "image", {'prompt': prompt})
        
    return result



# ==== CSV ANALYSIS TOOL ====
def summarize_csv(path, DEFAULT_MODEL, logger, memory=None):
    logger.info(f"Summarizing CSV... {path}")
    
    # Check if file exists
    assert os.path.exists(path), f"CSV file does not exist: {path}"
    
    # Check if we already have this analysis in memory
    if memory:
        # Get file from memory system if it exists
        file_records = memory.get_related_files(f"csv {os.path.basename(path)}", ["csv"])
        for file in file_records:
            if file.get("original_path") == path:
                logger.info(f"Using cached CSV summary for {path}")
                return file.get("content", '')
    
    df = pd.read_csv(path)
    summary = df.describe(include='all').to_string()
    
    # Extract key statistics for better memory retention
    stats = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            stats[column] = {
                'mean': float(df[column].mean()),
                'min': float(df[column].min()),
                'max': float(df[column].max()),
                'std': float(df[column].std())
            }
        
    # Convert stats to string for consistent storage
    stats_text = "\n\nKey Statistics:\n" + json.dumps(stats, indent=2)
    
    response = ollama.generate(
        model=DEFAULT_MODEL,
        prompt=f"Summarize the following CSV file: {path}\n\n{summary}{stats_text}",
    )
    result = response.response
    
    # Store in memory if available
    if memory:
        metadata = {'statistics': stats, 'columns': list(df.columns)}
        memory.add_file(path, result, "csv", metadata)
        
    return result



def describe_markdown(path, DEFAULT_MODEL, logger, memory=None):
    logger.info(f"Describing markdown... {path}")
    
    # Check if we already have this analysis in memory
    if memory:
        # Get file from memory system if it exists
        file_records = memory.get_related_files(f"markdown {os.path.basename(path)}", ["markdown"])
        for file in file_records:
            if file.get("original_path") == path:
                logger.info(f"Using cached markdown summary for {path}")
                return file.get("content", '')
    
    with open(path, "r") as f:
        content = f.read()
    response = ollama.generate(
        model=DEFAULT_MODEL,
        prompt=f"Summarize the following markdown file: {path}\n\n{content}",
    )
    result = response.response
    
    # Store in memory if available
    if memory:
        memory.add_file(path, result, "markdown", {'raw_content': content[:500]})
        
    return result