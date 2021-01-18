import gym
import numpy as np


from gym import spaces, error, utils
from gym.utils import seeding


class DiscreteAgent:
    def __init__(self, width):
        self.width = width
        self.pos = np.zeros((2,))

    def step(self, action):
        if action == 0:
            self.at = np.array([0,1])
        elif action == 1:
            self.at = np.array([-1, 0])
        elif action == 2:
            self.at = np.array([0,-1])
        else:
            self.at = np.array([1,0])
        self.pos = (self.pos + self.at)%self.width

class ContinuousAgent:
    def __init__(self, width):
        self.width = width
        self.pos = np.zeros((2, ))

    def step(self, action):
        self.at = action
        self.pos = (self.pos + self.at)%self.width


class GridWorldEnv(gym.Env):
    def __init__(self):
        super(GridWorldEnv).__init__()
        self._init = False
        self.steps = 0
        self.episode_reward = 0



    def init(self, low=0, high=100, agent_step=1, thrshld = 0, episode_step_limit=1000):
        self._init = True
        self.low = low
        self.high = high
        self.threshold = thrshld
        self.episode_step_limit = episode_step_limit
        assert self.high - self.low > agent_step
        rng = self.high - self.low
        self.createRandomGoal()
        if agent_step == 0:
            width = rng
            self.action_space = spaces.Box(0, 1, shape=(2,))
            self.agent_creator = ContinuousAgent
        else:
            width = rng//agent_step if rng%agent_step == 0 else (rng//agent_step) + 1
            self.action_space = spaces.Discrete(4)
            self.agent_creator = DiscreteAgent

        
        self.agent = self.agent_creator(width)
        self.observation_space = spaces.Box(low=low, high=high, shape=(2,))
        

    
    def createRandomGoal(self):
        self.goal_state = np.random.randint(self.low, self.high, size=(2,))
        def reward(st, action, st_next):
            return 1/((abs(self.goal_state[0] - st[0]) + abs(self.goal_state[1] - st[1])) + .01) # manhattan 
        self.reward = reward

    def set_goal_state(self, gs):
        self.goal_state = gs

    def set_reward(self,mapper):
        self.reward = mapper

    def reset(self):
        if not self._init:
            self.init() # init with default parameters

        self.agent = self.agent_creator(self.agent.width)
        self.episode_reward = 0
        self.steps = 0
        return self.agent.pos

    def step(self, action):
        self.steps += 1
        st = self.agent.pos
        self.agent.step(action)
        self.rt = self.reward(st, action, self.agent.pos)
        distance = np.linalg.norm(self.goal_state - self.agent.pos)
        
        if distance <= self.threshold or self.steps >= self.episode_step_limit:
            done = True
            self.rt += (self.episode_step_limit - self.steps)
        else:
            done = False
        
        self.episode_reward += self.rt
        return self.agent.pos, self.rt, done, {}

    def render(self, mode='human', close=False):
        print(f'Current Position: {self.agent.pos} \t Current Goal State: {self.goal_state}')
        print(f'Step: {self.steps} \t Action: {self.agent.at}')
        print(f'Episode Reward: {self.episode_reward}')
        print("--------------------------------------------------------------------------------------")

