from tools.uav_tools import *
from agent.agent import agent

# Construct the Tools agent
# agent_executor.invoke(
#     {"input": "Connect to Bebop2, takeoff and then rotate by z axis by 30 degrees for duration of 2 seconds"}
# )

messages = agent.invoke({"messages": [("human", "Connect to Bebop2, start the video feed, move the camera by tilt angle of 30 degress, and return the image array")]})
