import os
from langchain_groq import ChatGroq
     
llm_groq = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.1,
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=3,
    streaming=True,
)
