#!/usr/bin/env python3
from typing import List
import os
from fastapi import FastAPI
from tools.uav_tools import *
from langserve import add_routes
from agent.agent import agent_executor


app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(
    app,
    agent_executor
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8080)