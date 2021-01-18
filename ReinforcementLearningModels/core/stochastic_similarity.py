import gym
import numpy as np

from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def create_discrete_codebook(obs, dim, clusters):
    """"
    PARAMETERS:
    obs (np.ndarray (N, 298)): flattened observation array from running a model 
                                or from the experts
    dim (int): pca dimension
    clusters (int): Kmeans clusters

    RETURNS:
    codebook_states (np.ndarray (N, 1)): returns the states after being reduced and discretized
    """

    if(dim > 0):
        pca = PCA(dim)
        reduced_obs = pca.fit_transform(obs)
    else:
        reduced_obs = obs
    kmeans = KMeans(clusters)
    return kmeans.fit_predict(reduced_obs).reshape((-1,1))


def get_obs(predictor, env):
    """
     Observations over 30 episodes

     PARAMETERS:
     predictor: any model type or function to transform gym environment observations
     
     env (gym environment): the environment to get the observations from

    RETURNS:
    observations: a flattend array of observations
    lengths: a list of lengths for each episode
    """

    observations = []
    counts = []
    for _ in range(30):
        done = False
        o = env.reset()
        obs = [o]
        c = 0
        while not done:
            c += 1
            a = predictor(o)
            o, r, done, _ = env.step(a)
            obs.append(o)
        observations += obs
        counts.append(c)
    return np.array(observations), counts 


def lengths_to_idx(lengths):
    idx = [0]
    for l in lengths:
        idx.append(idx[-1] + l)
    return np.array(idx[1:-1]).astype(np.int)



def similarity(predictor1, predictor2, env):
    obs1, lengths1 = get_obs(predictor1, env)
    indices1 = lengths_to_idx(lengths1)
    codebook1 = create_discrete_codebook(obs1, -1, 12)
    agent1_observations = np.split(obs1, indices1)
    agent1_hmm = hmm.GaussianHMM(12, n_iter=25, verbose=False)
    agent1_hmm.fit(codebook1, lengths1)

    obs2, lengths2 = get_obs(predictor2, env)
    indices2 = lengths_to_idx(lengths2)
    codebook2 = create_discrete_codebook(obs2, -1, 12)
    agent2_observations = np.split(obs2, indices2)
    agent2_hmm = hmm.GaussianHMM(12, n_iter=25, verbose=False)
    agent2_hmm.fit(codebook2, lengths2)

    idx1 = np.random.choice(np.arange(len(lengths1)), size=(50,), replace=True)
    idx2 = np.random.choice(np.arange(len(lengths2)), size=(50,), replace=True)

    sim_score = []       
    for i,j in list(zip(idx1, idx2)):
        o1 = agent1_observations[i].reshape((-1, 1))
        o2 = agent2_observations[j].reshape((-1, 1))
        p11 = 10**(agent1_hmm.score(o1)/len(o1))
        p12 = 10**(agent1_hmm.score(o2)/len(o2))
        p21 = 10**(agent2_hmm.score(o1)/len(o1))
        p22 = 10**(agent2_hmm.score(o2)/len(o2))
        sim_score.append(np.sqrt((p12*p21)/(p11*p22)))


    return sum(sim_score)/len(sim_score)