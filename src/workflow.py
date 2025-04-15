from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from .state import AgentState
from src.llm.llm_interface import LLMInterface

llm = LLMInterface()
class GradeDocs(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")



class RAGWorkflow:
    def __init__(self, retriever_tool):
        self.workflow = StateGraph(AgentState)
        self.tools = [retriever_tool]
        self.retrieve = ToolNode([retriever_tool])
        self._setup_nodes()
        self._setup_edges()

    def _setup_nodes(self):
        self.workflow.add_node("agent", self._agent_node)
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node("generate", self._generator_node)
        
        self.workflow.add_node("rewrite", self._rewrite_node)

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
            self._grade_docs,
        )
        self.workflow.add_edge("generate", END)
        self.workflow.add_edge("rewrite", "agent")
        
    def compile(self):
        return self.workflow.compile()
    
    
    def _agent_node(self, state):
        print("---CALL AGENT---")
        messages = state["messages"]
            
        model = llm.bind_tools(self.tools)
        response = model.invoke(messages[0].content)
        return {"messages": [response]}


    def _generator_node(self, state):
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        docs = messages[-1].content

        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = prompt | llm | StrOutputParser()
            
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}



    def _rewrite_node(self, state):
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

        response = llm.generate_response(msg)
        return {"messages": [response]}



    def _grade_docs(self, state):
        print("---CHECK RELEVANCE---")
            
        llm_with_tool = llm.with_structured_output(GradeDocs)
            
        prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        chain = prompt | llm_with_tool

        messages = state["messages"]
        question = messages[0].content
        docs = messages[-1].content

        scored_result = chain.invoke({"question": question, "context": docs})
            
        if scored_result.binary_score == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return "generate"
        print("---DECISION: DOCS NOT RELEVANT---")
        return "rewrite"
        