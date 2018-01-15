import time
import gym
import gym_ple
import tensorflow as tf
import numpy as np

PREVIOUS_ACTION_MEMORY = 50
LEARNING_RATE = 0.05
RANDOM_ACTION_CHANCE = 0.1
REWARD_LAMBDA = 0.9

tf.set_random_seed(1)
np.random.seed(1)

env = gym.make('FlappyBird-v0')
env.reset()

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def run():
    x_ = tf.placeholder(tf.float32, shape=[None, 512, 288, 3])
    y_ = tf.placeholder(tf.float32, shape=[None])
    action_ = tf.placeholder(tf.int32)

    grayscale = tf.image.rgb_to_grayscale(x_)

    # Scale from [0,255] to [0,1]
    grayscale = grayscale * (1/255)

    w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 10], stddev=0.1))
    conv1 = conv2d(grayscale, w1)
    pool1 = max_pool_2x2(conv1)

    w2 = tf.Variable(tf.truncated_normal([5, 5, 10, 1], stddev=0.1))
    conv2 = conv2d(pool1, w2)
    pool2 = max_pool_2x2(conv2)
    

    # pool2.shape = (?, 128, 72, 1); 128*72*1 = 9216
    print(pool2.shape)
    flattened = tf.reshape(pool2, [-1,9216])

    w_dnn1 = tf.Variable(tf.truncated_normal([9216, 2], stddev=0.1))
    b_dnn1 = tf.Variable(tf.constant(0.1, shape=[2]))
    output = tf.tanh(tf.matmul(flattened, w_dnn1) + b_dnn1)
    # output = tf.sigmoid(tf.matmul(flattened, w_dnn1) + b_dnn1)
    # output = tf.matmul(flattened, w_dnn1) + b_dnn1

    cost = tf.reduce_mean((tf.gather(tf.split(output,2,axis=1), action_) - y_)**2)
    train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter('tensorboard/', sess.graph)

    state_history = np.array([])
    action_history = np.array([])
    reward_history = np.array([])

    for _ in range(1000000):
        # Get current screen array
        curr_state = env.render(mode='rgb_array')        
        
        # Determine next action
        prediction = sess.run(output, feed_dict={x_: [curr_state]})[0]
        action = None
        if np.random.random_sample() < RANDOM_ACTION_CHANCE:
            action = np.random.choice([0, 1]) # Occationally pick random action
        else:
            action = 0 if prediction[0] > prediction[1] else 1 # Big action with biggest predicted reward

        # Perform Action
        observation, reward, done, info = env.step(action)
        next_state = observation

        # Keep history of states
        state_history = np.array([*state_history, curr_state][-PREVIOUS_ACTION_MEMORY:])
        action_history = np.array([*action_history, action][-PREVIOUS_ACTION_MEMORY:])
        reward_history = np.array([*reward_history, reward][-PREVIOUS_ACTION_MEMORY:])

        if int(state_history.shape[0]) == PREVIOUS_ACTION_MEMORY: # Only perform descent once 10 actions have occured TODO: REMOVE
            # Calculate rewards
            if reward != 0:
                # Scale reward from [-5, 5] to [-1, 1]
                reward = (reward/5)
                if done:
                    y = np.array([reward]*int(state_history.shape[0]))
                else:
                    # implement look-back max rewards
                    y = np.array([reward]*int(state_history.shape[0]))


                print('Calcs:')
                # print(sess.run(cost, feed_dict={x_: state_history, y_: y, action_: action}))
                print('Action: ', action)
                print(y)
                print(sess.run(output, feed_dict={x_: state_history, y_: y, action_: action}))
                print(sess.run(tf.split(output,2,axis=1), feed_dict={x_: state_history, y_: y, action_: action}))
                print('Cost: ', sess.run(cost, feed_dict={x_: state_history, y_: y, action_: action}))

                # Perform train step
                sess.run(train, feed_dict={x_: state_history, y_: y, action_: action})

        if done:
            env.reset()
        # print(env.render(mode='rgb_array').shape)
        env.render(mode='human')

run()