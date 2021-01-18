"""
Extra tools 
"""


import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.utils import Sequence
from collections import defaultdict



def test_model(env, model, parameters):
    """
     Test a given environment with a (not necessarily) trained model under specific parameters

     env -> gym environment
     model -> trained stable_baselines model
     parameters -> parameters used to initialize training and training
    """
    
    testing_parameters = parameters['testing_parameters']
    
    # metrics
    reward_list = []
    episode_rewards = []
    episode_lengths = []
    num_episodes = 0
    eps_length = 0


    obs = env.reset()
    for _ in range(testing_parameters['iterations']):
        eps_length += 1
        action, _ = model.predict(obs)
        obs, rewards, dones, _ = env.step(action)
        reward_list.append(rewards)
        
        if(dones):
            episode_rewards.append(rewards)
            episode_lengths.append(eps_length)
            num_episodes += 1
            eps_length = 0
            # obs = env.reset()

        if(testing_parameters['render']):
            env.render()

    results = {}
    results['run_name'] = parameters['run_name']
    results.update(parameters['learning_parameters'])
    results.update(parameters['model_parameters'])
    results.update(testing_parameters)
    results['reward_average'] = np.mean(reward_list)
    results['num_episodes'] = num_episodes

    if(len(episode_lengths) > 0):
        results['avg_eps_len'] = np.mean(episode_lengths)
    else: 
        results['avg_eps_len'] = eps_length
    
    if(len(episode_rewards) > 0):
        results['avg_eps_rew'] = np.mean(episode_rewards)
    else:
        results['avg_eps_rew'] = rewards
    
    
    return results

class Ornstein_Uhlenbeck_Noise:
    def __init__(self, act_dim, mu = 0, theta = 0.1, sigma = 0.2):
        self.dim = act_dim
        self.mu = mu 
        self.theta = theta
        self.sig = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.dim) * self.mu

    def decay(self, d):
        self.theta *= d


    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sig + np.random.randn(len(x))
        self.state = x + dx
        return self.state

class Gaussian_Noise:
    def __init__(self, act_dim, loc=0, scale=1.0):
        self.mu = loc
        self.sig = scale
        self.dim = act_dim
    
    def reset(self):
        pass
    
    def decay(self, d):
        if(np.random.rand() > 0.5):
            self.sig *= d

    def noise(self):
        return np.random.normal(loc=self.mu, scale=self.sig, size=(self.dim))
        
class Zero_Noise:
    def __init__(self, act_dim, **kwargs):
        self.dim = act_dim
    def reset(self):
        pass
    def decay(self, d):
        pass
    def noise(self):
        return np.zeros(self.dim)

class summary:
    def __init__(self, writer):
        self.writer = writer


    def update(self, state):
        for k,v in state.items():
            if(type(v) == tuple):
                try:
                    x = tf.Summary(value=[tf.Summary.Value(tag = k, simple_value=v[0])])
                    self.writer.add_summary(x, v[1])
                except TypeError:
                    print(k)
        self.writer.flush()


def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def surrogate_loss(r, adv, prob, clip, c2):
    return -K.mean(K.minimum(r * adv, K.clip(r, min_value=1 - clip, max_value=1 + clip) * adv) + c2 * -(prob * K.log(prob + 1e-10)))

def ppo_loss(advantage, old_prediction, clip=0.2, c2=5e-3):
    def loss(y_true, y_pred):
        prob = K.sum(y_true * y_pred, axis=-1)
        old_prob = K.sum(y_true * old_prediction, axis=-1)
        r = prob/(old_prob + 1e-10)
        return surrogate_loss(r, advantage, prob, clip, c2)
    return loss

def ppo_loss_continuous(advantage, old_prediction, noise=1.0, clip=0.2, c2=5e-3):
    def loss(y_true, y_pred):
        var = K.square(noise)
        
        denom = K.sqrt(2 * var * np.pi)
        prob_num = K.exp( -K.square(y_true - y_pred)/(2*var))
        old_prob_num = K.exp( -K.square(y_true - old_prediction)/(2*var))
        prob = prob_num/denom
        old_prob = old_prob_num/denom
        # don't let old_prob be 0
        r = prob/(old_prob + 1e-10)
        return surrogate_loss(r, advantage, prob, clip, c2)
    return loss

def parse_events_file(path: str) -> pd.DataFrame:
    metrics = defaultdict(list)
    for e in tf.train.summary_iterator(path):
        for v in e.summary.value:
            if isinstance(v.simple_value, float):
                metrics[v.tag].append(v.simple_value)
    metrics_df = pd.DataFrame({k: v for k,v in metrics.items() if len(v) > 1})
    return metrics_df