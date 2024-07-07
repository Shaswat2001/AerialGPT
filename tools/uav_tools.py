from typing import Optional, Type
from pyparrot.Bebop import Bebop

from langchain.pydantic_v1 import BaseModel,Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import AsyncCallbackManagerForToolRun,CallbackManagerForToolRun

class UAVConnectTool(BaseTool):

    name = "UAVConnect"
    description = "Connects to the Parrot Bebop 2 and returns the bebop object"
    return_direct: bool = False

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None):
        
        bebop = Bebop()

        print("connecting")
        success = bebop.connect(10)

        if success:
            return bebop
        else:
            return None

    def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> None:
        
        return self._run(run_manager=run_manager.get_sync())

class UAVTakeOffinput(BaseModel):

    uavObj: Bebop = Field(description="Bebop class object for executing commands")

class UAVTakeOffTool(BaseTool):

    name = "UAVTakeOff"
    description = "Given a bebop object, send takeoff commands to the UAV"
    args_schema: Type[BaseModel] = UAVTakeOffinput
    return_direct: bool = False

    def _run(self, uavObj: Bebop, run_manager: Optional[CallbackManagerForToolRun] = None) -> None:
        
        uavObj.ask_for_state_update()
        uavObj.safe_takeoff(10)

    def _arun(self, uavObj: Bebop, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> None:
        
        self._run(uavObj, run_manager=run_manager.get_sync())