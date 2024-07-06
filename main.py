from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
from tools.uav_tools import *
from langchain_core.messages import HumanMessage
import os
from langgraph.prebuilt import create_react_agent

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

tools = [UAVConnectTool()]

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)

chat_model = ChatHuggingFace(llm=llm)

agent_executor = create_react_agent(chat_model, tools)

response = agent_executor.invoke(
    {"messages": [HumanMessage(content="Connect to bebop")]}
)
response["messages"]