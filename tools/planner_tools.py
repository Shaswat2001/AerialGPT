from typing import Optional, Type, Any, List
from pyparrot.Bebop import Bebop
import uuid

from langchain.pydantic_v1 import BaseModel,Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import AsyncCallbackManagerForToolRun,CallbackManagerForToolRun
from uav_tools import bebop_instances