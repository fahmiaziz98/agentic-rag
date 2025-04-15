import os
from langchain_google_genai import ChatGoogleGenerativeAI

class LLMInterface:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GEMINI_API_KEY"),
            model_name="gemini-2.0-flash",
            temperature=0.5,
            streaming=True,
            max_retries=3,
        )

    def generate_response(self, prompt: str):
        return self.llm.invoke(prompt)
