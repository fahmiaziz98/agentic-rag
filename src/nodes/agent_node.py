from .base_node import BaseNode

class AgentNode(BaseNode):
    def __init__(self, tools):
        super().__init__()
        self.tools = tools

    def __call__(self, state):
        print("---CALL AGENT---")
        messages = state["messages"]
        
        model = self.llm.bind_tools(self.tools)
        response = model.invoke(messages[0].content)
        return {"messages": [response]}