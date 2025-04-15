from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from .base_node import BaseNode

class GradeDocuments(BaseNode):
    class GradeDocs(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    def __call__(self, state) -> Literal["generate", "rewrite"]:
        print("---CHECK RELEVANCE---")
        
        llm_with_tool = self.llm.with_structured_output(self.GradeDocs)
        
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