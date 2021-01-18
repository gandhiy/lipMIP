
import os
import torch
import numpy as np

from copy import deepcopy
from pickle import load, dump
from os.path import join
from core.replay_experience import ReplayMemory

from pdb import set_trace as debug

class base:

    def __init__(
        self, env, name, path, tau, bs, g, 
        ms, vlg, wu, ve, ss, vb, tb, RANDOM_SEED):

        self.env = env
        self.v_env = deepcopy(env)
        self.model_name = name
        self.save_path = path
        self.__parameter_dict = None        
        self.tau = tau
        self.batch_size = bs
        self.gamma = g
        self.memory = ReplayMemory(capacity=ms)
        self.validation_logging = vlg
        self.warmup = wu
        self.save_log = ss
        self.num_validation_episode = ve
        self.state = {}
        self.verbose = vb
        self.tb_log = tb

        
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        self.env.seed(RANDOM_SEED)
        self.v_env.seed(RANDOM_SEED)



        self._parameter_dict = {}
        self.save_path = join(self.save_path, self.model_name)
        os.makedirs(self.save_path, exist_ok=True)

    def update_dictionary(self):
        if(len(self._parameter_dict)  == 0):
            self._parameter_dict = {
                key:value for key, value in self.__dict__.items() if 
                not key.startswith('__') and not callable(key) and key is not 'shap_predict'
            }
        else:
            for k,v in self.__dict__.items():
                if not k.startswith("__") and not callable(k) and k is not 'shap_predict':
                    self._parameter_dict[k] = v


    