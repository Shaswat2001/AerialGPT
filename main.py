from tools.uav_tools import *
from IPython.display import Image, display
from agent.agent import agent
from tools.utils import print_stream

# Construct the Tools agent
# agent_executor.invoke(
#     {"input": "Connect to Bebop2, takeoff and then rotate by z axis by 30 degrees for duration of 2 seconds"}
# )

inputs = {"messages": [("human", "Connect to Bebop2")]}
try:    
    print_stream(agent.stream(inputs, stream_mode="values"))
except KeyboardInterrupt:
    print("Landing the UAV")
    agent.invoke({"messages": [("human", "Land the UAV")]})