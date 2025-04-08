# Testing the Agno Integration

This guide explains how to test the Agno AI integration after applying our fixes.

## What Was Fixed

We resolved two key issues:

1. **API Key Handling**: The Google API key is now properly loaded from multiple sources and passed to the Agno agent.
2. **Response Capturing**: We've completely rewritten the response handling to properly capture output from any available Agno method.

## How to Test

### 1. Run the API Key Test Script

```bash
cd /data/SWATGenXApp/codes/AI_agent
python test_api_key.py
```

You should see a successful response and no "Empty response from agent" error.

### 2. Test Through the Web Interface

1. Restart the Flask application:
   ```bash
   cd /data/SWATGenXApp/codes/web_application
   # First kill any running instances
   kill $(ps aux | grep 'python run.py' | awk '{print $2}')
   # Start the server
   python run.py
   ```

2. Open the HydroGeo dataset page in your browser and test the chatbot.

### 3. Manual API Testing

```bash
# Test the initialize endpoint
curl -X POST http://localhost:5050/api/chatbot/initialize \
  -H "Content-Type: application/json" \
  -d '{"context": "hydrogeo_dataset", "model": "gemini-1.5-flash", "use_agno": true}'

# Test the chatbot endpoint
curl -X POST http://localhost:5050/api/chatbot \
  -H "Content-Type: application/json" \
  -d '{"context": "hydrogeo_dataset", "message": "Hello, tell me about PRISM data", "model": "gemini-1.5-flash", "use_agno": true}'
```

## Troubleshooting

If you encounter issues:

1. **Check the logs**:
   ```bash
   cd /data/SWATGenXApp/codes/web_application/logs
   tail -f flask-app-error.log
   ```

2. **Validate the Agno installation**:
   ```bash
   pip install --upgrade agno
   ```

3. **Analyze log messages**:
   - Look for messages starting with "Generated response using..."
   - Check which method successfully returned a response

4. **Debug the response capturing**:
   If response capturing is still an issue, you might need to inspect the raw output:
   ```python
   import agno
   print(f"Agno version: {agno.__version__}")
   ```

## Expected Behavior

After our fixes, you should see:

1. The API key being correctly loaded from config.py or environment
2. The Agno agent being created successfully
3. Responses being properly captured and returned to the frontend
4. No more "Empty response from agent" errors

## Methodologies Used

The enhanced response capture now tries multiple methods in this order:
1. `agent.run()` - Should return the string response directly
2. `agent.chat()` - Should return a response in newer Agno versions
3. `print_response()` with stdout capturing and regex parsing
4. Several fallback methods as a last resort

This robust approach ensures we can capture the response regardless of which Agno version or API pattern is being used. 