import gym
from matplotlib import animation
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1',render_mode='rgb_array')
obs = env.reset()

kp = 0.00
kv = -0.002
ka = -0.3
kav = -0.01
ks = -0.0000
sum_angle = 0.000
frames = []

def save_gif(frames):
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    anim.save('./CartPortCtrl.gif', writer='pillow', fps=30)

def CalcAction(obs):
    action = 0 # 0 meanleft, 1 means right
    global sum_angle
    sum= kp * (obs[0][0]) + kv * (obs[0][1]) + ka * (obs[0][2]) + kav * (obs[0][3]) + ks * sum_angle
    sum_angle += obs[0][2]
    if (sum < 0.0):
        action = 1
    else:
        action = 0
    return action

for _ in range(200):
    frames.append(env.render())
    action = CalcAction(obs)
    print('action = %d' % action)
    obs1, reward, done, info,_ = env.step(action)
    if (done):
        obs = env.reset()
env.close()
save_gif(frames)

