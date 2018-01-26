# Model
This model is an implementation of the one described in "Playing Atari with Deep Reinforcement Learning," with some modifications [1].

It uses a neural network with 3 convolutional layers and 2 fully connected layers to estimate the optimal Q function for any OpenAI gym environment.

# Steps to Reproduce
1. Install Necessary Requirements
2. Modify the Hyperparameters in gym-deep-q.py
    - Specify which environment with GYM_ENV
    - Specify a save directory with SAVE_DIR
    - If resuming training a model, specify the subdirectory inside SAVE_DIR with RESUME_SUB_DIR
    - If running a model to assess performance, set TRAIN to False
    - Note: Make sure any directories end in a '/'
3. Run gym-deep-q.py with python3

# Requirements
gym
ple - https://github.com/ntasfi/PyGame-Learning-Environment
gym_ple - https://github.com/lusob/gym-ple
tensorflow

# Other Notes
Some pre-trained models are located in the 'trained/' directory. They were run for arbitrary lengths of time and vary in fitness. You can run these models by setting SAVE_DIR to 'trained/' and RESUME_SUB_DIR to the subdirectory e.g. 'catcher/'

Because the Flappy Bird agent was trained with the goal of a universally-applicable model, the background was not removed, and thus the performance can vary drastically between runs, depending on the random backgrounds, pipe colors, and bird colors.

The strategy of filling the replay memory with some preliminary random actions was inspired by the GitHub repo at yenchenlin/DeepLearningFlappyBird [2].

# Sources
[1] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller. Playing Atari with Deep Reinforcement Learning. DeepMind Technologies, https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
[2] https://github.com/yenchenlin/DeepLearningFlappyBird
