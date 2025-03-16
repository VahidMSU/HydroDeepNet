from AI_agent.config import AgentConfig


class ModelSelector:
    """Utility class to select appropriate models for different tasks"""
    
    @staticmethod
    def get_model_for_task(task_name):
        """Get the appropriate model for a given task"""
        config = AgentConfig()
        
        # Mapping of task types to config model names
        task_model_mapping = {
            "query_analysis": config.query_analysis_model,
            "data_retrieval": config.data_processing_model,
            "analysis": config.analysis_model,
            "synthesis": config.synthesis_model,
            "climate_analysis": config.analysis_model,
            "crop_analysis": config.analysis_model,
            "pattern_recognition": config.models["medium"],  # Fixed: use models dict instead of medium_model
            "general": config.default_model
        }
        
        # Return the appropriate model or default if not found
        return task_model_mapping.get(task_name, config.default_model)
    
    @staticmethod
    def select_by_complexity(text_length, has_numbers=False):
        """Select model based on input complexity"""
        config = AgentConfig()
        
        if has_numbers and text_length > 500:
            return config.models["large"]  # Complex numerical analysis
        elif text_length > 1000:
            return config.models["medium"]  # Longer text processing
        elif has_numbers:
            return config.models["small"]  # Simple numerical tasks
        else:
            return config.models["tiny"]   # Very simple text tasks
