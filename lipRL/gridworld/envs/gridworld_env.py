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
        self._current_step = 0



    def init(self, low=0, high=100, step=1, thrshld = 0):
        self.low = low
        self.high = high
        self.threshold = thrshld
        self._episode_reward = 0
        assert self.high - self.low > step
        rng = self.high - self.low
        self.createRandomGoal()



        if step == 0:
            width = rng
            self.action_space = spaces.Box(0, 1, shape=(2,))
            self.agent_creator = ContinuousAgent
        else:
            width = rng//step if rng%step == 0 else (rng//step) + 1
            self.action_space = spaces.Discrete(4)
            self.agent_creator = DiscreteAgent

        
        self.agent = agent_creator(width)
        self.observation_space = spaces.Box(low=low, high=high, shape=(2,))
        

    

    
    def createRandomGoal(self):
        self.goal_state = np.random.randint(self.low, self.high, size=(2,))
        def reward(pos_t, action, pos_t1):
            return 1/(np.linalg.norm(self.goal_state - pos_t1) + 1.2) 
        self.reward = reward

    def set_goal_state(self, gs):
        self.goal_state = gs

    def set_reward(self,mapper):
        self.reward = mapper

    def reset(self):
        self.agent = self.agent_creator(self.agent.width)
        self._current_step = 0
        self._episode_reward = 0
        self.createRandomGoal()

        return self.agent.pos

    def step(self, action):
        st = self.agent.pos
        self.agent.step(action)
        self.at = self.agent.at
        distance = np.linalg.norm(self.goal_state - self.agent.pos)
        if distance <= self.threshold:
            done = True
        else:
            done = False
        self._step_reward = self.reward(self.agent.pos)
        self._episode_reward += self._step_reward
        return self.agent.pos, self._step_reward, done, {}

    def render(self, mode='human', close=False):
        print(f'Current Position: {self.agent.pos} \t Current Goal State: {self.goal_state}')
        print(f'Step: {self._current_step} \t Action: {self.at}')
        print(f'Episode Reward: {self._episode_reward}')
        print("--------------------------------------------------------------------------------------")

