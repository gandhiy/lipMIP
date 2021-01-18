import os
import sys
import torch
import numpy as np
import torch.nn as nn


sys.path.append("../..")

from time import time
from tqdm import tqdm
from copy import deepcopy
from .base import base
from os.path import join
from lipMIP import LipMIP
from hyperbox import Hyperbox
from relu_nets import ReLUNet
from core.replay_experience import Transition
from torch.utils.tensorboard import SummaryWriter as summary

class DQN_LipMIP(base):
    """
     Class for a DQN Agent compatibale with LipMIP. Initialize agent and then call
     agent.learn(episodes=N). Only works with discrete space 
     action environments.

     NON-DEFAULT ARGUMENTS
     * env(gym.env): Continuous action space environment
     
     * reward_class(reward_functions.reward): reward function to train DDPG, see models/reward_functions.py for more information

     * opt (torch.optim): optimizer for the DQN neural network

     DEFAULT ARGUMENTS
     * model_name (string): name to save model under (default -> 'temp')
     * batch_size(int): number of samples to train on (default -> 256)
     * memory_size(int): size of the replay buffer (default -> 50000)
     * gamma (float): discount factor (default -> 0.99)
     * tau (float): soft actor critic update parameter (default -> 0.001)
     * epsilon_min (float): minimum epsilon value for generating noise (default -> 0.001) 
     * epsilon_decay (float): decay rate for epsilon (default -> 0.995)
     * warmup (int): number of environment warmup episodes to set up replay buffer (default -> 25)
     * validation_logging (int): number of episodes between validation logging (default -> 25)
     * validation_episodes (int): number of episodes to validate on (default -> 5)
     * save_paths (string/file path): path to save model at (default -> .)
     * save_episodes (int): number of episodes between model saves (default -> 100)
     * layers (array): each element represents the number of nodes at layer i (default -> [64, 64])
     * regularization (string): choose between ['l1', 'l2', 'lip', 'na']
     * reg (float): L2 neural network layer regularization term (default = 0.01)
     * activation (string, keras.activations): the activation function for the neural network (default -> 'relu')
     * verbose (int): verbose output (default -> 0)
     * tb_log (boolean): whether to log information to the tensorboard (default -> True) #WIP
     * explainer_samples (int): number of samples to explain on from batch; must be less than or equal to batch size and -1 implies batch size (default -> -1)
     * RANDOM_SEED (int): default -> 1234

    """
    def __init__(
        self, env, opt, policy=None, layers=[64,64], regularization='l1', lam=1.0, model_name='temp', batch_size=256, memory_size=50000, gamma=0.995,
        tau = 0.001, epsilon_min=0.001, epsilon_decay=0.995, warmup=25, validation_logging=25, validation_episodes=5,
        save_paths='/Users/yashgandhi/Documents/lipRL/saved_models', save_episodes=100, verbose=0, tb_log=True, tb_log_step = 250, 
        tb_log_episode=10, RANDOM_SEED=1234, lipschitz_kwargs = {'num_threads': 2}):
        
        self.now = time()

        super(DQN_LipMIP, self).__init__(
            env, model_name, save_paths, tau, batch_size, 
            gamma, memory_size, validation_logging, warmup, validation_episodes,
            save_episodes, verbose, tb_log, RANDOM_SEED
        )

        if len(self.env.action_space.shape) > 0:
            self.num_actions = self.env.action_space.shape[0]
        else:
            self.num_actions = self.env.action_space.n
        
        self.adj_layers = layers
        self.adj_layers.insert(0, self.env.observation_space.shape[0]) # sticking to vector environments
        self.adj_layers.append(self.num_actions)

        if policy is None:
            self.critic = ReLUNet(layer_sizes=self.adj_layers)
            self.target = ReLUNet(layer_sizes=self.adj_layers)
            self.target.transfer_weights(self.critic, 1.0) #copy the weights over
        else:
            self.critic = policy
            self.target = deepcopy(self.critic) # custom models are required to have a transfer_weights function
            self.target.transfer_weights(self.critic, 1.0) # transfer_weights should copy weights from given model when tau = 1.0
        
        files = [f for f in os.listdir(self.save_path) if 'DQN' in f]
        self.save_path = join(self.save_path, f'DQNLip{len(files) + 1}')
        

        
        self.reg = regularization
        self.lam = max(0.0, min(1.0, lam))
        self.softmax_layer = nn.Softmax(dim=1)
        self.criterion = nn.MSELoss()
        self.optimizer = opt(self.critic.parameters(), lr=0.01)

        # logging functions/values
        if self.tb_log:
            logdir = join(self.save_path, 'tensorboard_logs')
            self.writer = summary(logdir) # going straight to tensorboard logger rather than using summary class in core.tools
        self.tb_log_step = tb_log_step
        self.tb_log_episode = tb_log_episode

        #init metrics
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay_factor = epsilon_decay
        self.__best_val_score = -np.inf
        self.running_loss = 0.0
        self.step = 1
        self.average_time_per_episode = 0

        # lipschitz regularization 
        self.dim = self.env.observation_space.shape[0]
        self.domain = Hyperbox.build_unit_hypercube(self.dim)
        self.c_vec = torch.Tensor(self.num_actions*[1.0])
        self.lipschitz_kwargs = lipschitz_kwargs
        

        print(f"Time to setup: {time() - self.now}")

    def regularization_techniques(self):
        print("Following Regularization Techniques: \n \'l1\' -> l1 weight penalty\
            \n \'l2\' --> l2 weight penalty \n \'lip\' --> lipschitz regularization")

        
    def _predict(self, obs, critic=True, detach=True):
        torch_obs = torch.Tensor(obs).type(torch.FloatTensor)
        if critic:
            tmp = self.softmax_layer.forward(self.critic.forward(torch_obs))
        else:
            tmp = self.softmax_layer.forward(self.target.forward(torch_obs))
        
        if detach:
            return tmp.detach().numpy()
        else:
            return tmp

    def predict(self, obs, critic=True):
        return np.argmax(self._predict(obs, critic), axis=1)
    
    def predict_max_proba(self,obs, critic=True):
        return np.amax(self._predict(obs, critic), axis=1)
    
    def environment_step(self, obs, done):
        if np.random.rand() > self.epsilon:
            at = self.predict(obs)[0]
        else:
            at = self.env.action_space.sample()

        obs_next, reward, done, info = self.env.step(at)
        trajectory = [obs, at, obs_next, done, reward] 
        obs = obs_next
        
        if done:
            self.count = 0
            obs = self.env.reset()
            trajectory[2] = obs
        self.memory.push(trajectory[0], trajectory[1], trajectory[2], trajectory[3], trajectory[4])
        return obs, done

    
    def batch_update(self):
        batch = Transition(*zip(*self.memory.sample(self.batch_size)))
        
        # create target output
        # y = r_t + max a' Q(s_{t+1}, a') if not done
        # y = r_t if done
        mask = np.ones(self.batch_size) * ([not l for l in  batch.done])
        y = self.gamma * self.predict_max_proba(batch.next_state)
        y *= mask
        y += batch.reward
        target = self._predict(batch.next_state, critic=False)
        target[np.arange(self.batch_size), batch.action] = y
        target = torch.Tensor(target).type(torch.FloatTensor)

        # fit section
        self.optimizer.zero_grad()
        output = self._predict(batch.state, critic=True, detach=False)
        loss = self.criterion(output, target)
        
        #regularization 
        if self.reg == 'l1':
            beta = torch.Tensor(0)
            for param in self.critic.parameters():
                beta += torch.norm(param, 1) ** 2
            loss += self.lam*beta
        elif self.reg == 'l2':
            beta = 0
            for param in self.critic.parameters():
                beta += torch.norm(param, 2) ** 2
            loss = loss + self.lam*beta
        elif self.reg == 'lip':
            results = LipMIP(
                self.critic,
                self.domain,
                self.c_vec,
                verbose=self.verbose > 0,
                **self.lipschitz_kwargs
            ).compute_max_lipschitz() 
            loss = loss + self.lam*results.value


        loss.backward()
        self.optimizer.step()
        
        # update metrics for writing summary        
        self.running_loss += loss.item()

    def learn(self, episodes=1000):
        obs = self.env.reset()
        self.total_episodes = episodes
        
        
        for e in tqdm(range(self.total_episodes)):
            self.now = time()
            self.current_episode = e + 1
            done = False
            while not done:
                self.step += 1
                obs, done = self.environment_step(obs, done)

                if self.memory.can_sample(self.batch_size):    
                    self.batch_update()
                    self.target.transfer_weights(self.critic, self.tau)
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay_factor

                if self.step%self.tb_log_step == 0:
                    self.tensorboard_update_step()

            if self.current_episode%self.tb_log_episode == 0:
                self.tensorboard_update_episode
            
            
            self.episode_time = time() - self.now
            self.average_time_per_episode += self.episode_time


    # plotting metrics
    def tensorboard_update_step(self):
        # write to board based on current step
        self.writer.add_scalar('training/training_loss', self.running_loss/self.tb_log_step, self.step)
        self.writer.add_scalar("training/epsilon", self.epsilon, self.step)

        # reset metrics
        self.running_loss = 0

    def tensorboard_update_episode(self):
        # write to board based on current episode
        self.writer.add_scaler('training/average_time_per_episode',self.average_time_per_episode/self.tb_log_episode, self.current_episode)
        self.writer.add_scaler('training/steps_per_episode', self.step/self.current_episode, self.current_episode)
        
        # reset metrics
        self.average_time_per_episode = 0


    def test(self, test_episodes=5):
        done = False
        test_env = deepcopy(self.env)
        obs = test_env.reset()
        rewards = []
        episode_reward = 0
        for te in range(test_episodes):
            print("---------------------------\n\n NEW EPISODE \n\n---------------------------")
            while not done:
                action = self.predict(obs)
                obs_t, rt, done, info = test_env.step(action)
                test_env.render()
                episode_reward += rt
                obs = obs_t
            if done:
                obs = test_env.reset()
                done = False
                rewards.append(episode_reward)
                episode_reward = 0

        return rewards
        


    # save/load mechanics (can be used to save best of/checkpoint models)
    def save(self, path):
        torch.save(self.critic, join(path, 'critic.pt'))
        torch.save(self.target, join(path, 'target.pt'))

    def load(self, path):
        self.critic = torch.load(join(path, 'critic.pt'))
        self.target = torch.load(join(path, 'target.pt'))
