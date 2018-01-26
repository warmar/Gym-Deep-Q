Steps to Reproduce:
    1. Install Necessary Requirements
    2. Modify the Hyperparameters in gym-deep-q.py
        - Specify which environment with GYM_ENV
        - Specify a save directory with SAVE_DIR
        - If resuming training a model, specify the subdirectory inside SAVE_DIR with RESUME_SUB_DIR
        - If running a model to assess performance, set TRAIN to False
        - Note: Make sure any directories end in a '/'
    3. Run gym-deep-q.py with python3

Requirements:
    gym
    ple - https://github.com/ntasfi/PyGame-Learning-Environment
    gym_ple - https://github.com/lusob/gym-ple
    tensorflow
