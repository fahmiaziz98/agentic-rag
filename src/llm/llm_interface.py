import os
from langchain_google_genai import ChatGoogleGenerativeAI

class LLMInterface:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini-2.0-flash",
            temperature=0.5,
            streaming=True,
            max_retries=3,
        )
    def with_structured_output(self, schema):
        return self.llm.with_structured_output(schema)
    
    def bind_tools(self, tools):
        return self.llm.bind_tools(tools)
    
    def generate_response(self, prompt: str):
        return self.llm.invoke(prompt)
