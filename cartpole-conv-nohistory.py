import time
import gym
import gym_ple
import tensorflow as tf
import numpy as np

LEARNING_RATE = 0.001
REWARD_GAMMA = 0.90
NUM_ACTIONS = 2

tf.set_random_seed(1)
np.random.seed(1)

env = gym.make('CartPole-v0')
env.reset()

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def run():
    x_ = tf.placeholder(tf.float32, shape=[400, 600, 3])
    y_ = tf.placeholder(tf.float32, shape=[NUM_ACTIONS])

    # Tensorflow uses (Num, Height, Width, Channels)
    image = [x_]

    # tf.summary.image('image', image)

    # Scale from [0,255] to [0,1]
    image = tf.multiply(image, (1/255))

    # Create computation graph
    num_channels = int(image.shape[3])

    w1 = tf.Variable(tf.truncated_normal([5, 5, num_channels, 10], stddev=0.1))
    tf.summary.histogram('w1', w1)
    conv1 = conv2d(image, w1)
    pool1 = max_pool_2x2(conv1)

    w2 = tf.Variable(tf.truncated_normal([5, 5, 10, 1], stddev=0.1))
    tf.summary.histogram('w2', w2)
    conv2 = conv2d(pool1, w2)
    pool2 = max_pool_2x2(conv2)
    
    num_image_pixels = 1
    for dimension in pool2.shape[1:]:
        num_image_pixels *= int(dimension)
    flattened = tf.reshape(pool2, [-1,num_image_pixels])

    w_dnn1 = tf.Variable(tf.truncated_normal([num_image_pixels, NUM_ACTIONS], stddev=0.1))
    tf.summary.histogram('w_dnn1', w_dnn1)
    b_dnn1 = tf.Variable(tf.constant(0.1))
    # output = tf.tanh(tf.matmul(flattened, w_dnn1) + b_dnn1)
    output = tf.matmul(flattened, w_dnn1) + b_dnn1

    output = tf.reshape(output, [NUM_ACTIONS])

    # Define cost function
    cost = tf.reduce_mean(tf.square(output - y_))
    tf.summary.scalar('cost', cost)

    # Define train step
    train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    # Run model
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    merged_summary = tf.summary.merge_all()
    saver = tf.train.Saver()

    summary_writer = tf.summary.FileWriter('cartpole/', sess.graph)

    next_state = env.render(mode='rgb_array')
    for _ in range(1000000):
        # Get current screen array
        curr_state = next_state
        
        # Determine next action
        predictions = sess.run(output, feed_dict={x_: curr_state})

        action = None
        random_chance = 1/((_/1000)+2) # Start random chance at 0.5 and slowly decrease
        if np.random.random_sample() < random_chance:
            action = np.random.choice([0, 1]) # Occationally pick random action
        else:
            action = 0 if predictions[0] > predictions[1] else 1 # Big action with biggest predicted reward

        # Perform Action
        observation, reward, done, info = env.step(action)
        next_state = env.render(mode='rgb_array')

        # Calculate reward
        y = predictions[:]
        if done:
            # End reward = reward
            y[action] = reward
        else:
            # Regular reward is current reward plus discounted max next reward
            next_predictions = sess.run(output, feed_dict={x_: next_state})
            y[action] = reward + REWARD_GAMMA*max(next_predictions[0], next_predictions[1])

        # DEBUG INFO:
        # print('predictions: ', predictions)
        # print('y: ', y)
        # print('action: ', action)
        # print('reward: ', reward)
        # print('Cost: ', sess.run(cost, feed_dict={x_state: curr_state, y_: y}))

        # Perform train step
        sess.run(train, feed_dict={x_: curr_state, y_: y})

        k = sess.run(merged_summary, feed_dict={x_: curr_state, y_: y})
        summary_writer.add_summary(k, global_step=_)

        # Save variables
        if _ % 1000 == 0:
            saver.save(sess, './cartpole/saves/save.chkp')

        env.render(mode='human')

        if done:
            env.reset()

run()