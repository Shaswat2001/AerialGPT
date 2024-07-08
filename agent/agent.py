from tools.uav_tools import *
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode,create_react_agent
from langgraph.graph import END, StateGraph, MessagesState
from langchain.agents import AgentExecutor,create_tool_calling_agent

tools = [UAVConnectTool(),UAVTakeOffTool(),UAVLandTool(),UAVDisplacementTool(),UAVRotationTool(),UAVVisionTool(),UAVSetParametersTool()]

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

system_message = "Welcome to AerialGPT: LLM based tool to control UAVs. How can I help you !"
# prompt_template = ChatPromptTemplate.from_messages([
#     ('system', system_template),
#     ("user", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])


# Construct the Tools agent
agent = create_react_agent(llm, tools,messages_modifier=system_message,debug=True)
# agent = create_tool_calling_agent(llm, tools, prompt_template)
# agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True,handle_parsing_errors=True)