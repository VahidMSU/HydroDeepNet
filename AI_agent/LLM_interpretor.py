import contextlib
import json
import requests
import sys
sys.path.append("/data/SWATGenXApp/codes/web_application/")
from app.utils import read_h5_file

# Define Ollama API server details
OLLAMA_SERVER = "http://35.9.219.76:5000"  # Update with your Ollama server's IP & port

def chat_with_deepseek(prompt):
    """
    Sends a POST request to the remote Ollama server and retrieves the response.
    curl -X POST http://35.9.219.76:5000/api/generate -H "Content-Type: application/json" -d '{"model": "deepseek-r1:70b", "prompt": "What is the capital of France?", "stream": false}'
    """
    if not prompt:
        return "Error: Empty prompt provided."
    try:
        print(f"Sending prompt to DeepSeek")
        response = requests.post(
            f"{OLLAMA_SERVER}/api/generate",
            json={"model": "deepseek-r1:70b", "prompt": prompt, "stream": True},
            timeout=60,  # Adjust timeout as needed
            stream=True
        )

        print(f"Received response from DeepSeek {response}")

        # Process streaming response
        response_text = ""
        for line in response.iter_lines():
            if line:
                with contextlib.suppress(json.JSONDecodeError):
                    json_line = json.loads(line.decode("utf-8"))
                    response_text += json_line.get("response", "")
        return response_text.strip()

    except requests.exceptions.RequestException as e:
        return f"Error communicating with DeepSeek: {e}"


def CDL_trend(lat_range, lon_range, years):
    """
    Extracts data for the given polygon across multiple years.
    """
    extracted_data = {}
    variable = "CDL"
    for year in years:
        subvariable = str(year)
        if data := read_h5_file(
            lat_range=lat_range,
            lon_range=lon_range,
            address=f"{variable}/{subvariable}",
        ):
            extracted_data[year] = data
    return extracted_data

if __name__ == "__main__":
    # Define spatial bounds for data extraction
    min_latitude, max_latitude = 43.658148, 44.164683
    min_longitude, max_longitude = -85.444332, -85.239256
    years_to_analyze = list(range(2010, 2022))  # Extract data from 2010 to 2021

    # Extract data for the given polygon over multiple years
    extracted_data = CDL_trend(
        lat_range=(min_latitude, max_latitude),
        lon_range=(min_longitude, max_longitude),
        years=years_to_analyze,
    )

    if not extracted_data:
        print("Error: No data found for any of the given years.")
        exit(1)

    # Generate a prompt for DeepSeek
    prompt = f"""
    You are an expert in analyzing land cover change over time.
    Here is crop classification data (CDL) for a bounding box in Michigan across multiple years from 2010 to 2022.
    Identify trends from 2010 to 2022 for major land-use changes, and any patterns observed:

    {json.dumps(extracted_data, indent=2)}

    Provide an in-depth analysis comparing different years.
    """

    # Query DeepSeek
    response = chat_with_deepseek(prompt)

    # Save response to file
    output_path = "/data/SWATGenXApp/codes/web_application/app/landcover_analysis.txt"
    with open(output_path, "w") as file:
        file.write(response)

    print(f"Analysis completed. Response saved to {output_path}")
