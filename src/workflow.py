from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from .state import AgentState

from .nodes.agent_node import AgentNode
from .nodes.generator_node import GeneratorNode
from .nodes.grader_node import GradeDocuments
from .nodes.rewrite_node import RewriteNode


class RAGWorkflow:
    def __init__(self, retriever_tool):
        self.workflow = StateGraph(AgentState)
        self.tools = [retriever_tool]
        self.retrieve = ToolNode([retriever_tool])
        self._setup_nodes()
        self._setup_edges()

    def _setup_nodes(self):
        self.workflow.add_node("agent", AgentNode(tools=[self.tools]))
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node("generate", GeneratorNode())
        self.workflow.add_node("grade", GradeDocuments())
        self.workflow.add_node("rewrite", RewriteNode())

    def _setup_edges(self):
        self.workflow.add_edge(START, "agent")
        self.workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "retrieve",
                END: END
            }
        )
        
        self.workflow.add_conditional_edges(
            "retrieve",
            self.workflow.get_node("grade"),
        )
        self.workflow.add_edge("generate", END)
        self.workflow.add_edge("rewrite", "agent")
        
    def compile(self):
        return self.workflow.compile()
        