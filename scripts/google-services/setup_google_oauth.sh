#!/bin/bash

# Script to help set up Google OAuth for the SWATGenX application

echo "Google OAuth Setup Helper for SWATGenX"
echo "====================================="
echo 
echo "This script will guide you through setting up Google OAuth credentials."
echo

# Step 1: Instructions for Google Cloud Console
echo "Step 1: Create OAuth credentials in Google Cloud Console"
echo "--------------------------------------------------------"
echo "1. Go to https://console.cloud.google.com/"
echo "2. Create a new project or select an existing one"
echo "3. Go to 'APIs & Services' > 'Credentials'"
echo "4. Click 'Create Credentials' > 'OAuth client ID'"
echo "5. Set up the OAuth consent screen if prompted"
echo "   - User Type: External"
echo "   - App Name: SWATGenX Application"
echo "   - User support email: Your email"
echo "   - Developer contact information: Your email"
echo "6. Create OAuth client ID:"
echo "   - Application type: Web application"
echo "   - Name: SWATGenX Web Client"
echo "   - Authorized JavaScript origins: "
echo "     * https://ciwre-bae.campusad.msu.edu (for production)"
echo "     * http://localhost:3000 (for development)"
echo "   - Authorized redirect URIs:"
echo "     * https://ciwre-bae.campusad.msu.edu/api/login/google/callback (production)"
echo "     * http://localhost:3000/api/login/google/callback (development)"
echo
echo "7. Click 'Create' and note your Client ID and Client Secret"
echo

# Step 2: Set up environment variables
echo "Step 2: Set up environment variables"
echo "----------------------------------"
echo "Now we'll help you set up the environment variables for your application."
echo

# Get environment variables
read -p "Enter your Google Client ID: " CLIENT_ID
read -p "Enter your Google Client Secret: " CLIENT_SECRET

# Create a .env file with the credentials
ENV_FILE="/data/SWATGenXApp/codes/web_application/.env"

if [ -f "$ENV_FILE" ]; then
    # Update existing file
    grep -v "GOOGLE_CLIENT" "$ENV_FILE" > "$ENV_FILE.tmp"
    echo "GOOGLE_CLIENT_ID=$CLIENT_ID" >> "$ENV_FILE.tmp"
    echo "GOOGLE_CLIENT_SECRET=$CLIENT_SECRET" >> "$ENV_FILE.tmp"
    mv "$ENV_FILE.tmp" "$ENV_FILE"
else
    # Create new file
    echo "GOOGLE_CLIENT_ID=$CLIENT_ID" > "$ENV_FILE"
    echo "GOOGLE_CLIENT_SECRET=$CLIENT_SECRET" >> "$ENV_FILE"
fi

echo
echo "Credentials have been saved to $ENV_FILE"
echo

# Step 3: Instructions for loading environment variables
echo "Step 3: Load environment variables"
echo "---------------------------------"
echo "Make sure to load these environment variables in your application:"
echo
echo "For development:"
echo "  source $ENV_FILE"
echo
echo "For production using Apache with mod_wsgi:"
echo "  Add these lines to your site configuration:"
echo "  SetEnv GOOGLE_CLIENT_ID $CLIENT_ID"
echo "  SetEnv GOOGLE_CLIENT_SECRET $CLIENT_SECRET"
echo
echo "Google OAuth setup complete! Restart your application to apply changes."
