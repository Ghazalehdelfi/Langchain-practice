from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Dict, Any
 
load_dotenv(override=True)

class Task(BaseModel):
    category: str = Field("category of the task")
    difficulty: str = Field("difficulty of the task")

    def to_dict(self) -> Dict[str, Any]:
        return {"category": self.category, "difficulty": self.difficulty}
    

task_parser = PydanticOutputParser(pydantic_object=Task)


def task(input:str) -> str:
    
    query = """The player wants to perform this task: {task}, 
        assign to this task one of the following categories:
         strength, dexterity, intelligence, wisdom, charisma
        and a level of diffculty between 1-20 based on how hard the task would be to complete (1 very easy - 20 very difficult)"""
    
    llm = ChatOpenAI(temperature=0.20,
        model_name="gpt-3.5-turbo")
    prompt = PromptTemplate(template=query, input_variables=["task"])

    chain = prompt | llm 
    result = chain.invoke(input={"task": input})

    return result


