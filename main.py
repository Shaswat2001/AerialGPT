from tools.uav_tools import *
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode,create_react_agent
from langgraph.graph import END, StateGraph, MessagesState
from langchain.agents import AgentExecutor,create_tool_calling_agent
from agent.agent import agent_executor

# Construct the Tools agent
agent_executor.invoke(
    {"input": "Connect to Bebop2, takeoff and then rotate by z axis by 30 degrees for duration of 2 seconds"}
)