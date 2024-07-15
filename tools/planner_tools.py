import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import matplotlib.pyplot as plt

from typing import Type,List
from langchain.pydantic_v1 import BaseModel,Field
from langchain_core.tools import BaseTool
from tools.uav_tools import bebop_instances
from tools.uav_tools import UAVDisplacementTool

class ToppraPlannerInput(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    instance_id: str = Field(description="Unique identifier for the Bebop instance")
    waypoints: List[List[float]] = Field(description="waypoints that the UAV needs to reach")
    vel_limit: List[List[float]] = Field(description="Velocity limit of the UAV [minimum,maximum]",default=[[-1, 1], [-1, 1], [-1, 1]])
    acc_limit: List[List[float]] = Field(description="Acceleration limit of the UAV [minimum,maximum]",default=[[-1, 1], [-1, 1], [-1, 1]])

class ToppraPlannerTool(BaseTool):

    name = "ToppraPlanner"
    description = "Given a list of waypoints, the velocity and acceleration limits, calculates the trajectory for the UAV"
    args_schema: Type[BaseModel] = ToppraPlannerInput
    return_direct: bool = False

    def _run(self,instance_id: str, waypoints: List[List[float]], vel_limit: List[List[float]] = [[-1, 1], [-1, 1], [-1, 1]], acc_limit: List[List[float]] = [[-1, 1], [-1, 1], [-1, 1]]) -> None:
                
        path = ta.SplineInterpolator(np.linspace(0, 1, num=len(waypoints)), waypoints)

        vlim = np.array([vel_limit])
        alim = np.array([acc_limit])
        pc_vel = constraint.JointVelocityConstraint(vlim)
        pc_acc = constraint.JointAccelerationConstraint(
        alim, discretization_scheme=constraint.DiscretizationType.Interpolation)

        instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper='seidel')
        jnt_traj = instance.compute_trajectory(0, 0)

        duration = jnt_traj.duration
        print("Found optimal trajectory with duration {:f} sec".format(duration))
        ts = np.linspace(0, duration, 100)
        qs = jnt_traj.eval(ts)
        qds = jnt_traj.evald(ts)
        qdds = jnt_traj.evaldd(ts)

        print(qs)

        for i in range(len(qs) - 1):

            wp = np.append(qs[i+1,:] - qs[i,:],0)
            UAVDisplacementTool._run(instance_id,wp)


import numpy as np
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import matplotlib.pyplot as plt

# Define a 3D path (waypoints)
way_pts = np.array([
    [0, 0, 0],
    [1, 2, 1],
    [2, 4, 3],
    [3, 6, 5],
    [4, 8, 6],
    [5, 10, 7]
])