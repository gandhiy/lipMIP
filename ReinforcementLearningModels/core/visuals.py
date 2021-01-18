import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation


def action_space_contour(env, i,j,model, discrete = False, figsize=(12,8)):
    high1 = min(env.observation_space.high[i], 1e8)
    high2 = min(env.observation_space.high[j], 1e8)
    low1 = max(env.observation_space.low[i], -1e8)
    low2 = max(env.observation_space.low[j], -1e8)


    x = np.linspace(low1, high1)
    y = np.linspace(low2, high2)
    mat = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    
    z = np.zeros((50**2, *env.observation_space.shape))
    z[:, i] = mat[:, 0]
    z[:, j] = mat[:, 1]
    out = model(z)

    fig = plt.figure(figsize=figsize, dpi=125)
    plt.grid()
    plt.xlabel(f"Observation {i}")
    plt.ylabel(f"Observation {j}")
    
    if discrete:
        for k in range(env.action_space.n):
            z_tmp = z[out == k]
            plt.scatter(z_tmp[:, i], z_tmp[:, j], label=f'Action {k}')
        plt.legend()
    else:
        if out.shape[1] > 1:
            pred = out[:, 0].reshape(50,50)
        else:
            pred = out.reshape(50,50)
        plt.contourf(x,y,pred)
        plt.colorbar()
    return fig

def run_model(env, num_epochs, predict): 
    images = []
    observations = []
    actions = []
    for e in range(num_epochs):
        ims = []
        acts = []
        reward = 0
        obs = env.reset()
        o = [obs]
        im = env.render(mode='rgb_array')
        ims.append(im)
        done = False
        while not done:
            a = predict(obs)
            acts.append(a)
            obs, rew, done, _ = env.step(a)
            o.append(obs)
            im = env.render(mode='rgb_array')
            ims.append(im)
            reward += rew
        print(f"Episode Reward = {reward}")
        images.append(ims)
        observations.append(o)
        actions.append(acts)
    env.close()
    del(env)
    return images, observations, actions

def animate_lines(*vals, xlim = None, ylim=None, frames = 1000):
    fig = plt.figure()
    ax = plt.axes(xlim=xlim, ylim=ylim)
    line, = ax.plot([], [], linewidth=1.5)
    

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = np.arange(i + 1)
        y = vals[0][:i + 1]
        line.set_data(x, y)

        return line,

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames = frames, interval=5, blit=True
    )

    anim.save("test_ani.mp4", fps = 30, extra_args=['-vcodec', 'libx264'])
    return anim

