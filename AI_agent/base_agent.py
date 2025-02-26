from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name, description):
        self.name = name
        self.description = description
        
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
