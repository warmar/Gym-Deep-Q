import time
import gym
import gym_ple

env = gym.make('FlappyBird-v0')
env.reset()

for _ in range(10000):
    if _ % 20 ==0:
        env.step(0)
    else:
        env.step(1)
    env.render(mode='human')
    # env.render(mode='rgb_array')
    time.sleep(0.1)