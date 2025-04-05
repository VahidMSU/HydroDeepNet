import json
import logging
import requests
from oauthlib.oauth2 import WebApplicationClient
from flask import current_app, redirect, url_for, request, session
from app.models import User
from app.extensions import db
from sqlalchemy.exc import OperationalError

# Configure logging
logger = logging.getLogger(__name__)

def get_google_provider_cfg():
    """Fetch Google's OAuth 2.0 configuration"""
    try:
        return requests.get(current_app.config['GOOGLE_DISCOVERY_URL']).json()
    except Exception as e:
        logger.error(f"Error fetching Google provider config: {e}")
        return None

def get_google_client():
    """Create and return a Google OAuth client"""
    return WebApplicationClient(current_app.config['GOOGLE_CLIENT_ID'])

def get_google_auth_url():
    """Generate the Google OAuth authorization URL"""
    client = get_google_client()
    google_provider_cfg = get_google_provider_cfg()
    
    if not google_provider_cfg:
        logger.error("Failed to fetch Google provider configuration")
        return None
        
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]
    
    # Construct the authorization URL
    return client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=f"{current_app.config['SITE_URL']}/api/login/google/callback",
        scope=["openid", "email", "profile"],
    )

def process_google_callback():
    """Process Google OAuth callback and return user info"""
    # Get authorization code from the callback request
    code = request.args.get("code")
    if not code:
        logger.error("No authorization code received from Google")
        return None
        
    client = get_google_client()
    google_provider_cfg = get_google_provider_cfg()
    
    if not google_provider_cfg:
        logger.error("Failed to fetch Google provider configuration")
        return None
        
    # Prepare the token request
    token_endpoint = google_provider_cfg["token_endpoint"]
    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=request.base_url,
        code=code
    )
    
    # Exchange the authorization code for a token
    try:
        token_response = requests.post(
            token_url,
            headers=headers,
            data=body,
            auth=(current_app.config['GOOGLE_CLIENT_ID'], current_app.config['GOOGLE_CLIENT_SECRET']),
        )
        
        # Parse the token response
        client.parse_request_body_response(json.dumps(token_response.json()))
    except Exception as e:
        logger.error(f"Error exchanging authorization code for token: {e}")
        return None
        
    # Use the token to fetch user info
    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
    uri, headers, body = client.add_token(userinfo_endpoint)
    
    try:
        userinfo_response = requests.get(uri, headers=headers, data=body)
        userinfo = userinfo_response.json()
        
        # Validate required fields
        if not userinfo.get("email_verified"):
            logger.warning(f"User email not verified: {userinfo.get('email')}")
            return None
            
        # Extract user info
        user_data = {
            "provider": "google",
            "provider_id": userinfo["sub"],
            "email": userinfo["email"],
            "name": userinfo.get("name"),
            "picture": userinfo.get("picture")
        }
        
        return user_data
    except Exception as e:
        logger.error(f"Error fetching user info: {e}")
        return None

def login_oauth_user(user_data):
    """Find or create user from OAuth data and log them in"""
    if not user_data or not user_data.get("email"):
        logger.error("Invalid OAuth user data")
        return None
        
    try:
        # Find or create user
        user = User.find_or_create_oauth_user(
            email=user_data["email"],
            provider=user_data["provider"],
            provider_id=user_data["provider_id"],
            name=user_data.get("name"),
            picture=user_data.get("picture")
        )
        
        return user
    except OperationalError as e:
        if "no such column: user.oauth_provider" in str(e):
            logger.error("Database schema missing OAuth columns. Running migration would fix this.")
            # Try running the migration directly
            try:
                logger.info("Attempting to run migration directly...")
                # Direct migration code
                import sqlite3
                db_path = '/data/SWATGenXApp/codes/web_application/instance/site.db'
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # First check if the user table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user'")
                if not cursor.fetchone():
                    logger.error("User table not found in database")
                    conn.close()
                    raise Exception("User table not found")
                
                # Check if the columns already exist
                cursor.execute("PRAGMA table_info(user)")
                columns = [column[1] for column in cursor.fetchall()]
                
                # Add each missing column if it doesn't exist
                added_columns = []
                oauth_cols = [
                    ('oauth_provider', 'VARCHAR(20)'),
                    ('oauth_id', 'VARCHAR(100)'),
                    ('oauth_name', 'VARCHAR(100)'),
                    ('oauth_picture', 'VARCHAR(255)')
                ]
                
                for col_name, col_type in oauth_cols:
                    if col_name not in columns:
                        logger.info(f"Adding column {col_name}")
                        cursor.execute(f"ALTER TABLE user ADD COLUMN {col_name} {col_type}")
                        added_columns.append(col_name)
                
                conn.commit()
                conn.close()
                
                if added_columns:
                    logger.info(f"Added columns: {', '.join(added_columns)}")
                    # Try again to create the user with OAuth info
                    return User.find_or_create_oauth_user(
                        email=user_data["email"],
                        provider=user_data["provider"],
                        provider_id=user_data["provider_id"],
                        name=user_data.get("name"),
                        picture=user_data.get("picture")
                    )
            except Exception as migration_error:
                logger.error(f"Migration failed: {migration_error}")
                
            # Try to find user by email only as fallback
            try:
                user = User.query.filter_by(email=user_data["email"]).first()
                if user:
                    logger.info(f"Found existing user by email: {user_data['email']}")
                    return user
                else:
                    # Try to create a basic user without OAuth fields
                    base_username = user_data["email"].split('@')[0]
                    username = base_username
                    counter = 1
                    while User.query.filter_by(username=username).first():
                        username = f"{base_username}{counter}"
                        counter += 1
                    
                    user = User(
                        username=username,
                        email=user_data["email"],
                        is_verified=True
                    )
                    db.session.add(user)
                    db.session.commit()
                    logger.info(f"Created new user with email: {user_data['email']}")
                    return user
            except Exception as inner_e:
                logger.error(f"Fallback user creation failed: {inner_e}")
                return None
        else:
            logger.error(f"Database error: {e}")
            return None
    except Exception as e:
        logger.error(f"Error creating/finding OAuth user: {e}")
        return None
