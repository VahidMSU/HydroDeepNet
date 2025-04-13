import yaml
import os
import logging
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config.yaml"

logger = logging.getLogger(__name__)

_config = None

def load_config() -> dict:
    """Loads configuration from YAML file and environment variables."""
    global _config
    if _config:
        return _config

    # Load from YAML file
    try:
        with open(CONFIG_FILE, 'r') as f:
            config_data = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {CONFIG_FILE}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {CONFIG_FILE}")
        # Provide a minimal default config to prevent crashes
        config_data = {
            'database_url': 'postgresql+psycopg://ai:ai@localhost:5432/ai',
            'db_tables': {},
            'log_dir': './logs',
            'base_report_dir': './Reports',
            'context_memory_db': 'memory_store.db',
            'knowledge_graph_json': 'knowledge_graph.json',
            'report_structure_json': 'report_structure.json',
            'default_model': 'gpt-4o',
            'image_model': 'gpt-4o',
            'query_analyzer_model': 'gpt-4o',
            'text_analyzer_model': 'gpt-4o',
            'spacy_model': 'en_core_web_sm',
            'text_analysis_truncate_chars': 5000,
        }
    except Exception as e:
        logger.error(f"Error loading configuration file {CONFIG_FILE}: {e}", exc_info=True)
        raise

    # Override with environment variables if they exist
    db_url_env = os.environ.get("PGVECTOR_CONNECTION")
    if db_url_env:
        config_data['database_url'] = db_url_env
        logger.info("Overriding database_url with PGVECTOR_CONNECTION environment variable.")

    # Add more environment variable overrides here if needed
    # Example:
    # api_key = os.environ.get("OPENAI_API_KEY")
    # if api_key:
    #     config_data['openai_api_key'] = api_key

    _config = config_data
    return _config

def get_config() -> dict:
    """Returns the loaded configuration dictionary."""
    if not _config:
        return load_config()
    return _config

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = get_config()
    print("Loaded Configuration:")
    import json
    print(json.dumps(cfg, indent=2))
    print("\nAccessing a specific value:")
    print(f"Database URL: {cfg.get('database_url')}")
    print(f"CSV Table Name: {cfg.get('db_tables', {}).get('csv')}") 