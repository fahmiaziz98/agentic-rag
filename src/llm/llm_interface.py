import os
from langchain_groq import ChatGroq

class LLMInterface:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.5,
            api_key=os.getenv("GROQ_API_KEY"),
            max_retries=3,
            streaming=True,
        )
    def with_structured_output(self, schema):
        return self.llm.with_structured_output(schema)
    
    def bind_tools(self, tools):
        return self.llm.bind_tools(tools)
    
    def generate_response(self, prompt: str):
        return self.llm.invoke(prompt)
