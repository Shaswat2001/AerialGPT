from typing import Optional, Type, Any
from pyparrot.Bebop import Bebop
import uuid

from langchain.pydantic_v1 import BaseModel,Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import AsyncCallbackManagerForToolRun,CallbackManagerForToolRun

bebop_instances = {}

class UAVConnectTool(BaseTool):

    name = "UAVConnect"
    description = "Connects to the Parrot Bebop 2 and returns the object of type Bebop() if the connection is successful"
    return_direct: bool = False

    def _run(self,input : str) -> str:
        
        global bebop_instances
        bebop = Bebop()

        print("connecting")
        success = bebop.connect(10)
        if success:
            bebop.smart_sleep(5)
            instance_id = str(uuid.uuid4())
            bebop_instances[instance_id] = bebop

            return instance_id

    def _arun(self,input : str) -> str:
        
        return self._run(input)

class UAVTakeOffinput(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    instance_id: str = Field(description="Unique identifier for the Bebop instance")

class UAVTakeOffTool(BaseTool):

    name = "UAVTakeOff"
    description = "Given a unique identifier (sent by UAVConnect), send takeoff commands to the UAV"
    args_schema: Type[BaseModel] = UAVTakeOffinput
    return_direct: bool = False

    def _run(self, instance_id: str) -> None:
        
        global bebop_instances
        bebop = bebop_instances.get(instance_id)

        if bebop:
            bebop.ask_for_state_update()
            bebop.safe_takeoff(10)
            bebop.smart_sleep(5)


    def _arun(self, uavObj: str) -> None:
        
        self._run(uavObj)

class UAVLandinput(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    instance_id: str = Field(description="Unique identifier for the Bebop instance")

class UAVLandTool(BaseTool):

    name = "UAVLand"
    description = "Given a unique identifier (sent by UAVConnect), send landing commands to the UAV"
    args_schema: Type[BaseModel] = UAVLandinput
    return_direct: bool = False

    def _run(self, instance_id: str) -> None:
        
        global bebop_instances
        bebop = bebop_instances.get(instance_id)

        if bebop:
            bebop.safe_land(10)

    def _arun(self, uavObj: str) -> None:
        
        self._run(uavObj)