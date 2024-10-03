from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub
from agents.taskagent import task
from agents.locationagent import location_information 


if __name__ == "__main__":
    while True:
        request = input()
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
        )
        template = """you are a game master, the player is asking you the following: {request}
        I want you to either tell the player what's the category and difficulty of the task they want to do if they ask to perform a specific task or if they are inquiring about the environment gather more information about the surroundings in the playbook and guide them accordingly"""

        prompt_template = PromptTemplate(
            template=template, input_variables=["request"]
        )
        tools_for_agent = [
            Tool(
                name="Generate task category and difficulty",
                func=task,
                description="useful for when you want to determine specific task's category and difficulty",
            ),
            Tool(
                name="Gather location information",
                func=location_information,
                description="useful for when you need to gather information about a location player wants to explore"
            )
        ]

        react_prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

        result = agent_executor.invoke(
            input={"input": prompt_template.format_prompt(request=request)}
        )
        print(result['output'])
