import sys
import os
import time

def interactive_document_reader():
    # Path configuration
    username = "admin"
    report_num = "20250324_222749"
    generated_report_path = f"/data/SWATGenXApp/Users/{username}/Reports/{report_num}/"
    
    # Import and initialize the reader
    from document_reader_core import InteractiveDocumentReader
    
    print("Initializing Interactive Document Reader...")
    reader = InteractiveDocumentReader()
    init_success = reader.initialize(auto_discover=True, base_path=generated_report_path)
    
    if not init_success:
        print("Error: Failed to initialize the document reader.")
        return
    
    print("Interactive Document Reader initialized. Type 'exit' to quit.")
    
    # Interactive loop
    while True:
        try:
            user_input = input("\nYour question: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
                
            # Process message
            response = reader.chat(user_input)
            print(f"\nAI: {response}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Let's continue.")

if __name__ == "__main__":
    interactive_document_reader()

    # Start chatting with the reader
    # response = reader.chat("Can you analyze the CSV file?")
    # print(response)