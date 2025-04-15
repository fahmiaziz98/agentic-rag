from abc import ABC, abstractmethod
from src.llm.llm_interface import LLMInterface

class BaseNode(ABC):
    def __init__(self):
        self.llm = LLMInterface()

    @abstractmethod
    def __call__(self, state):
        pass