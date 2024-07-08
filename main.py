from tools.uav_tools import *
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode,create_react_agent
from langgraph.graph import END, StateGraph, MessagesState
from langchain.agents import AgentExecutor,create_tool_calling_agent

tools = [UAVConnectTool(),UAVTakeOffTool(),UAVLandTool(),UAVDisplacementTool(),UAVRotationTool(),UAVVisionTool()]

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Construct the Tools agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True)
agent_executor.invoke(
    {"input": "Connect to Bebop2, takeoff and then rotate by z axis by 30 degrees for 2 seconds"}
)