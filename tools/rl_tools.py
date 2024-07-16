import numpy as np
import torch
from torch import nn 
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(nn.Module):

    def __init__(self,input_shape,n_action,bound):
        super(PolicyNetwork,self).__init__()


        self.bound = torch.tensor(bound,dtype=torch.float32).to(device)
        self.actionNet =  nn.Sequential(
            nn.Linear(input_shape,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,n_action),
            nn.Tanh()
        )

    def forward(self,state):

        action = self.actionNet(state)
        # self.bound = self.bound.reshape(action.shape)
        action = action*self.bound

        return action

class DDPG:
    '''
    DDPG Algorithm 
    '''
    def __init__(self,args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args # Argument values given by the user
        self.learning_step = 0 # counter to keep track of learning
        # Replay Buffer provided by the user

        self.reset()

    def choose_action(self,state,stage="training"):
        
        state = torch.Tensor(state).to(self.device)
        action = PolicyNetwork(state).to("cpu").detach().numpy()

        action = np.clip(action,self.args.min_action,self.args.max_action)

        return action

    def reset(self):

        self.TargetPolicyNetwork = self.policy(self.args.input_shape,self.args.n_actions,self.args.max_action).to(self.device)
    
    def load(self,env):

        print("--- LOADING NETWORK ---")
        self.TargetPolicyNetwork.load_state_dict(torch.load("tools/config/saves/training_weights/"+ env + "/ddpg_weights/TargetactorWeights.pth",map_location=torch.device('cpu')))

def build_parse():

    parser = argparse.ArgumentParser(description="RL Algorithm Variables")

    parser.add_argument("Environment",nargs="?",type=str,default="uam_vel_gazebo_tracking",help="Name of OPEN AI environment")
    parser.add_argument("input_shape",nargs="?",type=int,default=[],help="Shape of environment state")
    parser.add_argument("n_actions",nargs="?",type=int,default=[],help="shape of environment action")
    parser.add_argument("max_action",nargs="?",type=float,default=[],help="Max possible value of action")
    parser.add_argument("min_action",nargs="?",type=float,default=[],help="Min possible value of action")

    parser.add_argument("Algorithm",nargs="?",type=str,default="DDPG",help="Name of RL algorithm")
    parser.add_argument('tau',nargs="?",type=float,default=0.005)
    parser.add_argument('gamma',nargs="?",default=0.99)
    parser.add_argument('actor_lr',nargs="?",type=float,default=0.0001,help="Learning rate of Policy Network")
    parser.add_argument('critic_lr',nargs="?",type=float,default=0.0001,help="Learning rate of the Q Network")
    parser.add_argument('mult_lr',nargs="?",type=float,default=0.0001,help="Learning rate of the LAG constraint")

    parser.add_argument("mem_size",nargs="?",type=int,default=100000,help="Size of Replay Buffer")
    parser.add_argument("batch_size",nargs="?",type=int,default=64,help="Batch Size used during training")
    parser.add_argument("n_episodes",nargs="?",type=int,default=50000,help="Total number of episodes to train the agent")
    parser.add_argument("target_update",nargs="?",type=int,default=2,help="Iterations to update the target network")
    parser.add_argument("vision_update",nargs="?",type=int,default=5,help="Iterations to update the vision network")
    parser.add_argument("delayed_update",nargs="?",type=int,default=100,help="Iterations to update the second target network using delayed method")
    parser.add_argument("enable_vision",nargs="?",type=bool,default=False,help="Whether you want to integrate sensor data")
    
    # SOFT ACTOR PARAMETERS
    parser.add_argument("temperature",nargs="?",type=float,default=0.2,help="Entropy Parameter")
    parser.add_argument("log_std_min",nargs="?",type=float,default=np.log(1e-4),help="")
    parser.add_argument("log_std_max",nargs="?",type=float,default=np.log(4),help="")
    parser.add_argument("aux_step",nargs="?",type=int,default=8,help="How often the auxiliary update is performed")
    parser.add_argument("aux_epoch",nargs="?",type=int,default=6,help="How often the auxiliary update is performed")
    parser.add_argument("target_entropy_beta",nargs="?",type=float,default=-3,help="")
    parser.add_argument("target_entropy",nargs="?",type=float,default=-3,help="")

    # MISC VARIABLES 
    parser.add_argument("save_rl_weights",nargs="?",type=bool,default=True,help="save reinforcement learning weights")
    parser.add_argument("save_results",nargs="?",type=bool,default=True,help="Save average rewards using pickle")

    # USL 
    parser.add_argument("eta",nargs="?",type=float,default=0.05,help="USL eta")
    parser.add_argument("delta",nargs="?",type=float,default=0.1,help="USL delta")
    parser.add_argument("Niter",nargs="?",type=int,default=20,help="Iterations")
    parser.add_argument("cost_discount",nargs="?",type=float,default=0.99,help="Iterations")
    parser.add_argument("kappa",nargs="?",type=float,default=5,help="Iterations")
    parser.add_argument("cost_violation",nargs="?",type=int,default=20,help="Save average rewards using pickle")

    # Safe RL parameters
    parser.add_argument("safe_iterations",nargs="?",type=int,default=5,help="Iterations to run Safe RL once engaged")
    parser.add_argument("safe_max_action",nargs="?",type=float,default=[],help="Max possible value of safe action")
    parser.add_argument("safe_min_action",nargs="?",type=float,default=[],help="Min possible value of safe action")

    # Environment Teaching parameters
    parser.add_argument("safe_iterations",nargs="?",type=int,default=5,help="Iterations to run Safe RL once engaged")
    parser.add_argument("teach_alg",nargs="?",type=str,default="alp_gmm",help="How to change the environment")

    # Environment parameters List
    parser.add_argument("max_obstacles",nargs="?",type=int,default=10,help="Maximum number of obstacles need in the environment")
    parser.add_argument("obs_region",nargs="?",type=float,default=6,help="Region within which obstacles should be added")

    # ALP GMM parameters
    parser.add_argument('gmm_fitness_fun',nargs="?", type=str, default="aic")
    parser.add_argument('warm_start',nargs="?", type=bool, default=False)
    parser.add_argument('nb_em_init',nargs="?", type=int, default=1)
    parser.add_argument('min_k', nargs="?", type=int, default=2)
    parser.add_argument('max_k', nargs="?", type=int, default=11)
    parser.add_argument('fit_rate', nargs="?", type=int, default=250)
    parser.add_argument('alp_buffer_size', nargs="?", type=int, default=500)
    parser.add_argument('random_task_ratio', nargs="?", type=int, default=0.2)
    parser.add_argument('alp_max_size', nargs="?", type=int, default=None)

    args = parser.parse_args("")

    return args

def setup_agent():

    args1 = build_parse()
    args2 = build_parse()

    args1.state_size = 9
    args1.input_shape = 9
    args1.n_actions = 3
    args1.max_action = np.array([0.1,0.1,0.1])
    args1.min_action = -np.array([0.1,0.1,0.1])

    args2.state_size = 606
    args2.input_shape = 606
    args2.n_actions = 3
    args2.max_action = np.array([0.1,0.1,0.1])
    args2.min_action = -np.array([0.1,0.1,0.1])

    agent1 = DDPG(args = args1)
    agent2 = DDPG(args = args2)

    agent1.load("uam_vel_gazebo_tracking_3d")
    agent2.load("uam_vel_gazebo_obs_pcd")

    return agent1,agent2
