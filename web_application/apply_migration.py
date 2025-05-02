#!/usr/bin/env python3
"""
Standalone script to apply the OAuth database migration.
Run this script directly to add OAuth columns to the database.
"""
import os
import sys
import sqlite3
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/data/SWATGenXApp/codes/web_application/logs/migration.log')
    ]
)
logger = logging.getLogger("MigrationRunner")

def create_migrations_directory():
    """Ensure the migrations directory exists"""
    migrations_dir = '/data/SWATGenXApp/codes/web_application/migrations'
    if not os.path.exists(migrations_dir):
        logger.info(f"Creating migrations directory: {migrations_dir}")
        os.makedirs(migrations_dir, exist_ok=True)
    return migrations_dir

def create_oauth_migration_file():
    """Create the OAuth migration file if it doesn't exist"""
    migrations_dir = create_migrations_directory()
    migration_file = os.path.join(migrations_dir, 'add_oauth_columns.py')
    
    if os.path.exists(migration_file):
        logger.info(f"Migration file already exists: {migration_file}")
        return migration_file
    
    logger.info(f"Creating OAuth migration file: {migration_file}")
    
    # Define the migration script content
    migration_content = '''"""
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
'''
    
    # Write the migration script to the file
    with open(migration_file, 'w') as f:
        f.write(migration_content)
    
    logger.info(f"Created migration file: {migration_file}")
    return migration_file

def run_migration_direct():
    """Run the database migration directly without importing"""
    try:
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

def main():
    """Main function to run the migration"""
    logger.info("Starting OAuth database migration")
    
    # First try to run the migration directly
    logger.info("Attempting direct database migration...")
    direct_result = run_migration_direct()
    
    if direct_result:
        logger.info("✅ Direct migration succeeded!")
        print("✅ OAuth columns migration completed successfully!")
        print("The OAuth login feature should now work correctly.")
        return 0
    
    # If direct migration fails, try with the migration file
    logger.info("Direct migration failed, trying with migration file...")
    migration_file = create_oauth_migration_file()
    
    try:
        # Add migrations directory to the Python path
        migrations_dir = os.path.dirname(migration_file)
        if migrations_dir not in sys.path:
            sys.path.append(os.path.dirname(migration_file))
        
        # Import and run the migration
        from add_oauth_columns import run_migration
        result = run_migration()
        
        if result:
            logger.info("✅ Migration from file succeeded!")
            print("✅ OAuth columns migration completed successfully!")
            print("The OAuth login feature should now work correctly.")
            return 0
        else:
            logger.error("❌ Migration from file failed!")
            print("❌ Migration failed. Check the logs for details.")
            return 1
            
    except ImportError:
        logger.error("❌ Failed to import the migration module.")
        print("❌ Failed to import the migration module.")
        print("Please check if the migrations directory exists and contains the add_oauth_columns.py file.")
        return 1
    except Exception as e:
        logger.error(f"❌ An error occurred during migration: {str(e)}")
        print(f"❌ An error occurred during migration: {str(e)}")
        return 1

if __name__ == "__main__":
    # Make sure log directory exists
    os.makedirs('/data/SWATGenXApp/codes/web_application/logs', exist_ok=True)
    
    sys.exit(main())
