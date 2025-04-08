# Google API Key Configuration Guide

This guide helps you configure and test the Google API key for the Agno integration in the HydroGeo Assistant.

## Configuration Options

The application will look for the Google API key in the following places (in order):

1. Environment variable `GOOGLE_API_KEY`
2. Flask application config (`current_app.config['GOOGLE_API_KEY']`)
3. The `Config` class in `web_application/config.py`
4. A default fallback value (not recommended for production)

## Verifying Your Configuration

We've added several tools to help you verify your API key configuration:

### 1. Run the API Key Test Script

```bash
cd /data/SWATGenXApp/codes/AI_agent
python test_api_key.py
```

This script will:
- Check if the API key is in your environment
- Verify that `agno_hydrogeo.py` can retrieve the key
- Test creating an Agno agent
- Test generating a simple response

### 2. Check Application Logs

After starting your Flask application, check for these log messages:

```bash
cd /data/SWATGenXApp/codes/web_application/logs
grep -i "api_key" flask-app-error.log
```

You should see messages like:
- "Exported GOOGLE_API_KEY to environment"
- "Set GOOGLE_API_KEY in environment"
- "Got API key from Flask config" or "Got API key from config.py"

### 3. Test the API Directly

Make a test request to the chatbot API endpoint:

```bash
curl -X POST http://localhost:5050/api/chatbot/initialize \
  -H "Content-Type: application/json" \
  -d '{"context": "hydrogeo_dataset", "model": "gemini-1.5-flash", "use_agno": true}'
```

## Troubleshooting

If you're still having issues with the API key:

1. **Verify the key value** - Make sure your API key is valid and has access to the Gemini API

2. **Check environment variables** - Run `echo $GOOGLE_API_KEY` to see if it's set

3. **Restart the application** - Changes to configuration files require a restart:
   ```bash
   cd /data/SWATGenXApp/codes/web_application
   kill $(ps aux | grep 'python run.py' | awk '{print $2}')
   python run.py
   ```

4. **Set the key manually** - Before starting the application:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   cd /data/SWATGenXApp/codes/web_application
   python run.py
   ```

5. **Check the API response** - If the API returns an error about authentication, your key may have:
   - Expired
   - Insufficient permissions
   - Usage limits exceeded
   - Invalid format

## Further Configuration

If you need to use a different API key in different environments:

1. **Development**: Set in `.env` file or directly in terminal
2. **Production**: Set in system environment or through deployment configuration

## Making Changes to the Configuration

If you need to update your API key:

1. Edit `web_application/config.py` to update the `GOOGLE_API_KEY` value
2. Restart the application to apply changes
3. Run the test script to verify the new key is working 