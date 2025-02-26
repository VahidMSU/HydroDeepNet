import contextlib
import subprocess
import requests
import json
import numpy as np
from cdl_trend import cdl_trends
from prism import PRISM_Dataset
import geopandas 
from config import AgentConfig
from model_selector import ModelSelector

# Function to check if Ollama is running without using CPU
def is_ollama_running():
    try:
        response = requests.head("http://localhost:11434/api/tags", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def chat_with_deepseek(prompt, model=None, task=None):
    """
    Chat with language model, using the appropriate model for the task complexity
    
    Args:
        prompt (str): The prompt text
        model (str, optional): Specific model to use
        task (str, optional): Task type for automatic model selection
    """
    if not is_ollama_running():
        return "Error: Ollama is not running. Start it using 'ollama serve'."
        
    # Select appropriate model based on input parameters
    if model is None:
        if task:
            model = ModelSelector.get_model_for_task(task)
        else:
            # Analyze prompt complexity
            has_numbers = any(char.isdigit() for char in prompt)
            model = ModelSelector.select_by_complexity(len(prompt), has_numbers)
    
    print(f"Using model: {model} for prompt length: {len(prompt)}")
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": True},
            timeout=60,
            stream=True,
        )
        print(f"Model response status: {response.status_code}")  
        
        if response.status_code != 200:
            print(f"Error response content: {response.content.decode('utf-8')}")
            return f"Error: Received status code {response.status_code} from DeepSeek API."

        response_text = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_line = json.loads(line.decode("utf-8"))
                    if "response" in json_line:
                        response_text += json_line["response"]
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line as JSON: {line}")
                    continue
        
        return response_text.strip() or "No valid response received"

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with DeepSeek: {e}")
        return f"Error communicating with DeepSeek: {e}"

def run_gdal_command(command):
    """Run a GDAL command using subprocess and return the output."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"GDAL Error: {e.stderr.strip()}"

def get_bounding_box(county,state):
    
    path = AgentConfig.USGS_governmental_path

    gdf = geopandas.read_file(path, layer="GU_CountyOrEquivalent")
    county_shape = gdf[(gdf["STATE_NAME"] == state) & (gdf["COUNTY_NAME"] == county)].to_crs("EPSG:4326")
    bbox = county_shape.total_bounds.tolist()
    min_lon, min_lat, max_lon, max_lat = bbox

    print(f"Bounding box for {county} County, {state} (EPSG:4326): {min_lon, min_lat, max_lon, max_lat}")

    return min_lon, min_lat, max_lon, max_lat


def chunk_years(years, chunk_size=3):
    """Split years into smaller chunks."""
    return [years[i:i + chunk_size] for i in range(0, len(years), chunk_size)]

def analyze_year_chunk(extracted_data, years, pr_data, tmax_data, tmin_data):
    """Analyze a specific chunk of years with both land cover and climate data."""
    chunk_data = {year: extracted_data[year] for year in years}
    year_indices = [y - min(years) for y in years]
    
    avg_rainfall = np.mean([pr_data[i] for i in year_indices])
    avg_tmax = np.mean([tmax_data[i] for i in year_indices])
    avg_tmin = np.mean([tmin_data[i] for i in year_indices])
    
    prompt = f"""
    You are an expert in analyzing agricultural patterns and their relationship with climate conditions.
    Analyze the following data for a region in Michigan for years {min(years)} to {max(years)}:

    1. Land Cover Classification Data:
    {chunk_data}

    2. Climate Conditions:
    - Average Annual Precipitation: {avg_rainfall:.2f} mm
    - Average Maximum Temperature: {avg_tmax:.2f}°C
    - Average Minimum Temperature: {avg_tmin:.2f}°C

    Please analyze:
    1. Major land use patterns and changes
    2. Potential relationships between climate conditions and land use
    3. Notable agricultural trends influenced by weather patterns
    
    Provide practical insights on how climate might have influenced farming decisions.
    """
    
    return chat_with_deepseek(prompt, task="analysis")

if __name__ == "__main__":

    min_lon, min_lat, max_lon, max_lat = get_bounding_box("Mecosta", "Michigan")
    print(f"Bounding box for the extracted County, Michigan (EPSG:4326): {min_lon, min_lat, max_lon, max_lat}")

    if min_lon:
        config = {
            "RESOLUTION": 250,
            "huc8": None,
            "video": False,
            "aggregation": "annual",
            "start_year": 2008,
            "end_year": 2011,
            'bounding_box': [min_lon, min_lat, max_lon, max_lat],
        }
    else:
        config = {
            "RESOLUTION": 250,
            "huc8": None,
            "video": False,
            "aggregation": "annual",
            "start_year": 2008,
            "end_year": 2011,
            'bounding_box': [-85.444332, 43.658148, -85.239256, 44.164683],
        }

    prism_dataset = PRISM_Dataset(config)
    pr_prism, tmax_prism, tmin_prism = prism_dataset.get_spatial_average_over_time()

    print(f"PRISM data shape: {pr_prism.shape}, {tmax_prism.shape}, {tmin_prism.shape}")
    print("PRISM data:")
    print(f"Precipitation: {pr_prism}")
    print(f"Max Temperature: {tmax_prism}")
    print(f"Min Temperature: {tmin_prism}")

    extracted_data = cdl_trends(config)

    if not extracted_data:
        print("Error: No data found for any of the given years.")
        exit(1)

    years_to_analyze = np.arange(config['start_year'], config['end_year'] + 1)
    year_chunks = chunk_years(years_to_analyze)
    chunk_analyses = []

    for chunk in year_chunks:
        chunk_indices = [y - config['start_year'] for y in chunk]
        analysis = analyze_year_chunk(
            extracted_data,
            chunk,
            pr_prism[chunk_indices],
            tmax_prism[chunk_indices],
            tmin_prism[chunk_indices]
        )
        chunk_analyses.append(f"\nAnalysis for years {chunk}:\n{analysis}")

    final_prompt = f"""
    Review the following detailed analyses of land cover changes and climate patterns in Michigan 
    from {config['start_year']} to {config['end_year']}. Provide a comprehensive summary that:
    1. Identifies long-term trends in both land use and climate patterns
    2. Explains potential causal relationships between weather patterns and agricultural choices
    3. Suggests implications for future land use planning in this region

    Previous analyses:
    {' '.join(chunk_analyses)}
    """

    final_analysis = chat_with_deepseek(final_prompt)
    print(f"\nFinal Summary:\n{final_analysis}")
    complete_report = '\n\n'.join(chunk_analyses + ["\nFinal Summary:\n" + final_analysis])

    output_path = "landcover_analysis.txt"
    with open(output_path, "w") as file:
        file.write(complete_report)

    print(f"Analysis completed. Response saved to {output_path}")