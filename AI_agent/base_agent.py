from abc import ABC, abstractmethod
from model_selector import ModelSelector

class BaseAgent(ABC):
    def __init__(self, name, description, task_type=None):
        self.name = name
        self.description = description
        self.task_type = task_type or name.lower()
        self.model = ModelSelector.get_model_for_task(self.task_type)
        
    @abstractmethod
    def process(self, input_data, context=None):
        """Process input data and return results"""
        pass

    def format_message(self, role, content):
        return {
            "role": role,
            "content": content,
            "agent": self.name
        }
        
    def get_model(self, complexity=None):
        """Get the appropriate model for this agent's tasks"""
        if complexity == "high":
            return ModelSelector.get_model_for_task("analysis")
        elif complexity == "low":
            return ModelSelector.get_model_for_task("data_retrieval")
        return self.model
