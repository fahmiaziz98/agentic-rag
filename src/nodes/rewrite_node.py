from .base_node import BaseNode
from langchain_core.messages import HumanMessage

class RewriteNode(BaseNode):
    def __call__(self, state):
        print("---REWRITE---")
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f""" \n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """,
            )
        ]

        response = self.llm.generate_response(msg)
        return {"messages": [response]}