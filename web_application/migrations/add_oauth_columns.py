"""
Migration script to add OAuth-related columns to the User table.

This script safely adds the missing OAuth columns to the existing database.
"""
import sqlite3
import os
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OAuth-Migration")

def run_migration():
    """Adds the OAuth-related columns to the User table if they don't exist"""
    try:
        # Connect to the database
        db_path = '/data/SWATGenXApp/codes/web_application/instance/site.db'
        logger.info(f"Connecting to database at {db_path}")
        
        if not os.path.exists(db_path):
            logger.error(f"Database file not found at {db_path}")
            return False
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # First check if the user table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user'")
        if not cursor.fetchone():
            logger.error("User table not found in database")
            conn.close()
            return False
        
        # Check if the columns already exist
        cursor.execute("PRAGMA table_info(user)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Define the columns to add with their types
        columns_to_add = {
            'oauth_provider': 'VARCHAR(20)',
            'oauth_id': 'VARCHAR(100)',
            'oauth_name': 'VARCHAR(100)',
            'oauth_picture': 'VARCHAR(255)'
        }
        
        # Track if we added any columns
        columns_added = False
        
        # Add each missing column if it doesn't exist
        for column_name, column_type in columns_to_add.items():
            if column_name not in columns:
                logger.info(f"Adding column {column_name} ({column_type})")
                try:
                    cursor.execute(f"ALTER TABLE user ADD COLUMN {column_name} {column_type}")
                    columns_added = True
                except sqlite3.OperationalError as e:
                    logger.error(f"Error adding column {column_name}: {e}")
                    # Continue with other columns even if one fails
        
        if not columns_added:
            logger.info("All OAuth columns already exist, no changes needed")
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        logger.info("Migration completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Running OAuth columns migration...")
    success = run_migration()
    if success:
        print("✅ Migration completed successfully!")
        print("The OAuth login feature should now work correctly.")
    else:
        print("❌ Migration failed. Check the logs for details.")
        sys.exit(1)
