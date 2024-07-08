from typing import Optional, Type, Any, List
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
from .utils import *
import numpy as np
import uuid

from langchain.pydantic_v1 import BaseModel,Field
from langchain_core.tools import BaseTool

bebop_instances = {}

class UAVConnectinput(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    input: str = Field(description="Name of the UAV")

class UAVConnectTool(BaseTool):

    name = "UAVConnect"
    description = "Connects to the Parrot Bebop 2 and returns the object of type Bebop() if the connection is successful"
    args_schema: Type[BaseModel] = UAVConnectinput
    return_direct: bool = False

    def _run(self,input : str) -> str:
        
        global bebop_instances
        bebop = Bebop()

        print("CONNECTING TO BEBOP")
        
        try:
            success = bebop.connect(10)
            bebop.smart_sleep(5)
            instance_id = str(uuid.uuid4())
            bebop_instances[instance_id] = bebop

            return instance_id
        
        except Exception as e:
            print("Connection Failed : {e}")

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

class UAVDisplacementinput(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    instance_id: str = Field(description="Unique identifier for the Bebop instance")
    displacement: List[float] = Field(description="List discribing the displacement of UAV in X (forward or backward), Y (right or left) and Z (up and down) axis")

class UAVDisplacementTool(BaseTool):

    name = "UAVDisplacement"
    description = "Given a unique identifier (sent by UAVConnect) and list containing the displacement vector, send movement commands to the UAV"
    args_schema: Type[BaseModel] = UAVDisplacementinput
    return_direct: bool = False

    def _run(self, instance_id: str, displacement: List[float]) -> None:
        
        global bebop_instances
        bebop = bebop_instances.get(instance_id)

        if bebop:
            bebop.move_relative(displacement[0],displacement[1],displacement[2])

    def _arun(self, instance_id: str, displacement: List[float]) -> None:
        
        self._run(instance_id, displacement)

class UAVSetParametersinput(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    instance_id: str = Field(description="Unique identifier for the Bebop instance")
    max_altitude: float = Field(description="The maximum allowable altitude of the UAV in meters",default=None)
    max_distance: float = Field(description="The max distance between the takeoff and the UAV in meters",default=None)
    max_tilt: float = Field(description="The max allowable tilt in degrees for the UAV",default=None)
    max_tilt_rotation_speed: float = Field(description="The max allowable tilt rotation speed in degree/s",default=None)
    max_vertical_speed: float = Field(description="The maximum allowable vertical speed in m/s",default=None)
    max_rotation_speed: float = Field(description="The maximum allowable rotation speed in degree/s",default=None)

class UAVSetParametersTool(BaseTool):

    name = "UAVDisplacement"
    description = "Given a unique identifier (sent by UAVConnect) and values of different parameters to set on the UAV, default is None for all the variables"
    args_schema: Type[BaseModel] = UAVSetParametersinput
    return_direct: bool = False

    def _run(self, instance_id: str, max_altitude: float = None, max_distance: float = None, max_tilt: float = None, max_tilt_rotation_speed: float = None, max_vertical_speed: float = None, max_rotation_speed: float = None) -> None:
        
        global bebop_instances
        bebop = bebop_instances.get(instance_id)

        if bebop:
            if max_altitude:
                bebop.set_max_altitude(max_altitude)
            if max_distance:
                bebop.set_max_distance(max_distance)
            if max_tilt:
                bebop.set_max_tilt(max_tilt)
            if max_tilt_rotation_speed:
                bebop.set_max_tilt_rotation_speed(max_tilt_rotation_speed)
            if max_vertical_speed:
                bebop.set_max_vertical_speed(max_vertical_speed)
            if max_rotation_speed:
                bebop.set_max_rotation_speed(max_rotation_speed)

class UAVRotationinput(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    instance_id: str = Field(description="Unique identifier for the Bebop instance")
    rotation: List[float] = Field(description="List discribing the rotation of UAV in X (Roll), Y (Pitch), Z (Yaw) axis in radians, vertical movement and duration in seconds")

class UAVRotationTool(BaseTool):

    name = "UAVRotation"
    description = "Given a unique identifier (sent by UAVConnect) and list containing the rotation vector, send movement commands to the UAV"
    args_schema: Type[BaseModel] = UAVRotationinput
    return_direct: bool = False

    def _run(self, instance_id: str, rotation: List[float]) -> None:
        
        global bebop_instances
        bebop = bebop_instances.get(instance_id)

        if bebop:
            bebop.fly_direct(rotation[0],rotation[1],rotation[2],rotation[3],rotation[4])

    def _arun(self, instance_id: str, rotation: List[float]) -> None:
        
        self._run(instance_id, rotation)

class UAVVisioninput(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    instance_id: str = Field(description="Unique identifier for the Bebop instance")

class UAVVisionTool(BaseTool):

    name = "UAVVision"
    description = "Given a unique identifier (sent by UAVConnect) and returns the image from camera as a numpy array"
    args_schema: Type[BaseModel] = UAVVisioninput
    return_direct: bool = False

    def _run(self, instance_id: str) -> np.ndarray:
        
        global bebop_instances
        bebop = bebop_instances.get(instance_id)

        bebopVision = DroneVision(bebop, is_bebop=True)

        userVision = UserVision(bebopVision)
        bebopVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
        success = bebopVision.open_video()
        image = np.array([0])
        if (success):
            
            image = userVision.image
            bebopVision.close_video()

        return image

    def _arun(self, instance_id: str) -> np.ndarray:
        
        self._run(instance_id)