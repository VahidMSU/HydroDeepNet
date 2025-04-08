# Agno Integration for HydroGeo Assistant

This integration adds [Agno](https://docs.agno.com/) powered capabilities to the HydroGeo Assistant, enabling more advanced AI features using Google's Gemini models.

## Overview

The HydroGeo Assistant now leverages Agno's lightweight agent framework to power conversations about environmental and hydrological data. This integration provides:

- Advanced natural language understanding and generation
- Model selection capabilities (currently supporting Gemini 1.5 Flash)
- Markdown formatting for more readable responses
- Graceful fallback to legacy implementation if Agno is unavailable

## Setup

1. Install Agno:
   ```bash
   pip install agno
   ```

2. Set up the Google API key:
   ```bash
   # Add to your .bashrc or .bash_profile
   export GOOGLE_API_KEY="your-google-api-key"
   ```

3. Enable Agno integration (enabled by default):
   ```bash
   # Add to your .bashrc or .bash_profile
   export USE_AGNO="true"
   ```

4. **IMPORTANT**: Restart the web application to apply the changes:
   ```bash
   # Navigate to the web application directory
   cd /data/SWATGenXApp/codes/web_application
   
   # For development server
   kill $(ps aux | grep 'python run.py' | awk '{print $2}')
   python run.py
   
   # For production with gunicorn
   sudo systemctl restart swatgenx-webapp
   # or
   kill -HUP $(ps aux | grep gunicorn | grep swatgenx | awk '{print $2}')
   ```

## Recent Updates

**[2025-04-08]** Fixed API compatibility issue:
- The Agno API has changed, and the `Agent` object no longer has a `respond` method
- Updated `agno_hydrogeo.py` to use `print_response` instead, with fallbacks to other methods
- Added robust error handling to ensure the assistant works even with API changes
- The application will now gracefully fall back to alternative methods if the primary one fails

## Implementation Approach

We've taken a simplified integration approach:

1. **Direct Integration**: The Agno functionality is integrated directly into the existing `chatbot.py` file, eliminating the need for a separate routes module.

2. **Smart Fallbacks**: The implementation intelligently falls back to legacy behavior if:
   - Agno is not installed
   - The Google API key is not available
   - The specified model is not supported
   - Any runtime errors occur with Agno

3. **Client Control**: The frontend can explicitly request Agno or non-Agno responses by setting the `use_agno` parameter in requests.

4. **Import Safety**: All imports are designed to be safe and gracefully handle errors, preventing application crashes if dependencies are missing.

## Testing

Run the test script to verify that the Agno integration is working correctly:

```bash
cd /data/SWATGenXApp/codes/AI_agent
python test_agno.py
```

You can also test the interactive chat functionality:

```bash
# Test with Agno (if available)
python interactive_AI.py

# Test the fallback mode without Agno
python interactive_AI.py --no-agno
```

## Files

- `agno_hydrogeo.py`: Main integration file that provides functions for creating and using Agno agents
- `test_agno.py`: Test script to verify the integration works correctly
- `interactive_AI.py`: Command-line interface for testing the Agno integration
- `web_application/app/chatbot.py`: Flask blueprint that handles AI assistant requests with Agno integration

## Usage in Web Application

The HydroGeo Assistant frontend has been updated to use the Agno integration automatically. Users can:

1. Select a model from the dropdown (currently Gemini 1.5 Flash is available, with more models coming soon)
2. See the connection status to the Agno framework
3. Interact with the assistant using natural language
4. Receive responses with proper markdown formatting

## Implementation Details

The integration consists of three main components:

1. **Backend API Routes** (`/api/chatbot` endpoints): 
   - Accept requests from the frontend
   - Connect to the Agno framework if available
   - Fall back to legacy behavior if needed
   - Return formatted responses

2. **Agno Wrapper** (`agno_hydrogeo.py`):
   - Creates and manages Agno agents
   - Handles model selection and validation
   - Processes responses

3. **Frontend Component** (`HydroGeoAssistant.js`):
   - Provides UI for interacting with the assistant
   - Handles model selection
   - Displays connection status
   - Renders markdown responses

## Troubleshooting

- **API Key Issues**:
  - Check if your GOOGLE_API_KEY is properly set in your environment
  - Try exporting the key manually before starting the application: `export GOOGLE_API_KEY="your-key"`

- **Package Dependencies**:
  - Make sure Agno is installed in the correct Python environment: `pip install agno`
  - Check for any dependency conflicts using `pip check`

- **API Version Compatibility**:
  - If you see errors about missing methods (e.g., `'Agent' object has no attribute 'respond'`),
    make sure you're using a compatible version of Agno
  - Try updating to the latest version: `pip install --upgrade agno`
  - Check the Agno documentation for API changes: https://docs.agno.com/

- **Application Context Errors**:
  - If the application fails to start with errors about "working outside of application context",
    check that `chatbot.py` and other modules are not using Flask's `current_app` outside of request handlers

- **Import Errors**:
  - If you see import errors related to Agno, make sure the path to AI_agent is correct
  - Try adding the path manually: `export PYTHONPATH=$PYTHONPATH:/data/SWATGenXApp/codes`

- **Logging**: 
  - Check the application logs for detailed error information
  - For gunicorn: `/data/SWATGenXApp/codes/web_application/logs/gunicorn-error.log`
  - For Flask development server: Standard output in terminal

- **Environment Variable**: 
  - Make sure the USE_AGNO environment variable is set to "true" if you want to use Agno
  - Check with `echo $USE_AGNO`

## Future Enhancements

1. **Multi-Modal Support**: Add image upload and analysis capabilities
2. **Knowledge Integration**: Connect the assistant to domain-specific knowledge bases
3. **Multi-Agent Teams**: Build specialized agents for different environmental data domains
4. **New Models**: Add support for additional models like Claude 3.5 Sonnet 