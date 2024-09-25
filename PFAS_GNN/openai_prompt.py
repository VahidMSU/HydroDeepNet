import openai
import os

# Set the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

def chat_with_gpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use 'gpt-4' for the latest model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            n=1,  # Number of responses to return, must be 1
            stop=None, # Stop sequence
            temperature=0.7,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    # Example logging output
    log_file_path = "GNN_gw_pfas.txt"
    with open(log_file_path, "r") as file:
        logging_output = file.read()

    prompt = f"Here is some logging output from GNN machine learning experiment. provide a breakdonw of the workflow and results. Be analytical and pay attention to the numbers\n{logging_output}"
    response = chat_with_gpt(prompt)
    with open("GNN_gw_pfas_interpretation.txt", "w") as file:
        file.write(response)
