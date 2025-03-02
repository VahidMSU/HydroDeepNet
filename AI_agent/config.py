
class AgentConfig:
    """Configuration settings for the AI Agent."""
    
    # Model settings
    MODEL_NAME = "gpt-4"
    TEMPERATURE = 0.7
    MAX_TOKENS = 2000
    
    # Report generation settings
    REPORT_TYPES = {
        "summary": "Brief summary of model results",
        "detailed": "Detailed analysis with metrics and visualizations",
        "technical": "Technical report with all model parameters and results"
    }
    
    DEFAULT_REPORT_TYPE = "summary"
    
    # File paths
    TEMPLATES_DIR = "/data/SWATGenXApp/codes/AI_agent/templates"
    OUTPUT_DIR = "/data/SWATGenXApp/codes/AI_agent/reports"
    
    # API settings
    API_TIMEOUT = 60  # seconds
    RETRY_ATTEMPTS = 3
    
    @classmethod
    def get_report_settings(cls, report_type=None):
        """Get specific settings for a report type."""
        if report_type is None:
            report_type = cls.DEFAULT_REPORT_TYPE
            
        if report_type not in cls.REPORT_TYPES:
            raise ValueError(f"Invalid report type: {report_type}")
            
        return {
            "type": report_type,
            "description": cls.REPORT_TYPES[report_type],
            "template": f"{cls.TEMPLATES_DIR}/{report_type}_template.md"
        }
