# src/nodes/generator_node.py
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from .base_node import BaseNode

class GeneratorNode(BaseNode):
    def __call__(self, state):
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        docs = messages[-1].content

        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = prompt | self.llm | StrOutputParser()
        
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}