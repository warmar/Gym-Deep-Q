import time
import gym
import gym_ple
import tensorflow as tf

PREVIOUS_ACTION_MEMORY = 100

env = gym.make('FlappyBird-v0')
env.reset()

state_history = []
action_history = []
reward_history = []

for _ in range(100):
    action = None
    if _ % 20 ==0:
        action = 0
    else:
        action = 1
    observation, reward, done, info = env.step(action)

    state_history.append(observation)
    action_history.append(action)
    reward_history.append(reward)
    
    print(observation.shape)
    # print(env.render(mode='rgb_array').shape)
    print(done)
    print(reward)
    env.render(mode='human')
    time.sleep(0.025)