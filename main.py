from langchain_openai import ChatOpenAI
from tools.uav_tools import *
from langchain_core.messages import HumanMessage
import os
from langchain.agents import AgentExecutor,create_tool_calling_agent
from langgraph.prebuilt import ToolNode,create_react_agent
from langgraph.graph import END, StateGraph, MessagesState
from langchain_core.prompts import ChatPromptTemplate


tools = [UAVConnectTool(),UAVTakeOffTool(),UAVLandTool()]

from langchain_openai import ChatOpenAI

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
    {"input": "Connect to Bebop2 and then takeoff and then land"}
)


# llm_with_tools = llm.bind_tools(tools)
# response = llm_with_tools.invoke("Connect to Bebop 2 and then takeoff")
# agent = create_react_agent(llm, tools)

# agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)

# response = agent.invoke(
#     {"messages": [HumanMessage(content="Connect to parrot bebop 2")]}
# )

# response = agent_executor.invoke(
#     {"input": "Connect to parrot bebop "}
# )

# print(response)

# tool_node = ToolNode(tools)

# chat_model = ChatHuggingFace(llm=llm).bind_tools(tools=tools)

# # Define the function that calls the model
# def call_model(state: MessagesState):
#     messages = state['messages']
#     response = chat_model.invoke(messages)
#     # We return a list, because this will get added to the existing list
#     return {"messages": [response]}

# def should_continue(state: MessagesState) -> Literal["tools", END]:
#     messages = state['messages']
#     last_message = messages[-1]
#     # If the LLM makes a tool call, then we route to the "tools" node
#     if last_message.tool_calls:
#         return "tools"
#     # Otherwise, we stop (reply to the user)
#     return END


# # Define a new graph
# workflow = StateGraph(MessagesState)

# # Define the two nodes we will cycle between
# workflow.add_node("agent", call_model)
# workflow.add_node("tools", tool_node)

# # Set the entrypoint as `agent`
# # This means that this node is the first one called
# workflow.set_entry_point("agent")

# # We now add a conditional edge
# workflow.add_conditional_edges(
#     # First, we define the start node. We use `agent`.
#     # This means these are the edges taken after the `agent` node is called.
#     "agent",
#     # Next, we pass in the function that will determine which node is called next.
#     should_continue
# )

# workflow.add_edge("tools", 'agent')
# app = workflow.compile()

# # Use the Runnable
# final_state = app.invoke(
#     {"messages": [HumanMessage(content="Connect to parrot bebop 2")]},
#     config={"configurable": {"thread_id": 42}}
# )

# print(final_state)
# print(final_state["messages"][-1].content)