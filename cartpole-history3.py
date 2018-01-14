import os
import gym
import gym_ple
import tensorflow as tf
import numpy as np

GYM_ENV = 'CartPole-v0'
SAVE_DIR = './cartpole/'
LEARNING_RATE = 0.001
REWARD_GAMMA = 0.95
NUM_POSSIBLE_ACTIONS = 2
RANDOM_ACTION_RATE = 1000
HISTORY_MAX_SIZE = 10000
HISTORY_RAND_SAMPLE_SIZE = 50
TRAIN = True

# Create subfolder for each separate run
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
runs = os.listdir(SAVE_DIR)
i = 0
while str(i) in runs:
    i += 1
SAVE_DIR = SAVE_DIR + '%d/' % i

# Set random seeds for consistency
tf.set_random_seed(1)
np.random.seed(1)

# Create and reset environment
env = gym.make(GYM_ENV)
env.reset()
image_shape = env.render(mode='rgb_array').shape


# Define model functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, w, stride):
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def run():
    raw_images = tf.placeholder(tf.float32, shape=[None, *image_shape])
    # Convert to grayscale
    processed_images = tf.image.rgb_to_grayscale(raw_images)

    # Scale down
    processed_images = tf.image.resize_images(processed_images, [100, 100])

    # Scale from [0,255] to [0,1]
    processed_images = tf.multiply(processed_images, (1 / 255))

    # Create computation graph
    x_ = tf.placeholder(tf.float32, shape=[None, *(processed_images.shape[1:])])
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_POSSIBLE_ACTIONS])

    tf.summary.image('x_image', x_)

    num_channels = int(x_.shape[3])

    # Convolution + Pool Layer 1
    with tf.name_scope('conv-pool1'):
        w_conv1 = weight_variable([8, 8, num_channels, 32])
        b_conv1 = bias_variable([32])
        tf.summary.histogram('w_conv1', w_conv1)
        tf.summary.histogram('b_conv1', b_conv1)
        conv1 = tf.nn.relu(conv2d(x_, w_conv1, 4) + b_conv1)
        pool1 = max_pool_2x2(conv1)

    # Convolution + Pool Layer 2
    with tf.name_scope('conv-pool2'):
        w_conv2 = weight_variable([4, 4, 32, 32])
        b_conv2 = bias_variable([32])
        tf.summary.histogram('w_conv2', w_conv2)
        tf.summary.histogram('b_conv2', b_conv2)
        tf.summary.histogram('w2', w_conv2)
        conv2 = tf.nn.relu(conv2d(pool1, w_conv2, 4) + b_conv2)
        pool2 = max_pool_2x2(conv2)

    # Flatten image
    num_image_pixels = 1
    for dimension in pool2.shape[1:]:
        num_image_pixels *= int(dimension)
    flattened = tf.reshape(pool2, [-1, num_image_pixels])

    # Fully Connected Layer
    with tf.name_scope('fc1'):
        w_fc1 = weight_variable([num_image_pixels, 512])
        b_fc1 = bias_variable([512])
        tf.summary.histogram('w_fc1', w_fc1)
        tf.summary.histogram('b_fc1', b_fc1)
        fc1 = tf.nn.relu(tf.matmul(flattened, w_fc1) + b_fc1)

    with tf.name_scope('fc2'):
        w_fc2 = weight_variable([512, NUM_POSSIBLE_ACTIONS])
        b_fc2 = bias_variable([NUM_POSSIBLE_ACTIONS])
        tf.summary.histogram('w_fc2', w_fc2)
        tf.summary.histogram('b_fc2', b_fc2)
        output = tf.matmul(fc1, w_fc2) + b_fc2

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
    if not TRAIN:
        saver.restore(sess, SAVE_DIR + 'saves/save.chkp')
    else:
        summary_writer = tf.summary.FileWriter(SAVE_DIR, sess.graph)

    history_d = None

    score = 0
    next_state = env.render(mode='rgb_array')  # Allows us to render the screen only once per step
    next_state = sess.run(processed_images, feed_dict={raw_images: [next_state]})[0]
    for _ in range(1000000):
        # Get current screen array
        curr_state = next_state

        # Determine next action
        predictions = sess.run(output, feed_dict={x_: [curr_state]})[0]

        action = None
        random_chance = 1 / ((_ / RANDOM_ACTION_RATE) + 2)  # Start random chance at 0.5 and slowly decrease
        if np.random.random_sample() < random_chance:
            action = np.random.choice([0, 1])  # Occationally pick random action
        else:
            action = 0 if predictions[0] > predictions[1] else 1  # Big action with biggest predicted reward

        # Perform Action
        observation, reward, done, info = env.step(action)
        next_state = env.render(mode='rgb_array')
        next_state = sess.run(processed_images, feed_dict={raw_images: [next_state]})[0]
        score += reward

        if TRAIN:
            # Update histories
            if not done:
                transition = [curr_state, action, reward, next_state]
            else:
                transition = [curr_state, action, reward, None]

            if history_d is not None:
                history_d = np.vstack((history_d, [transition]))
            else:
                history_d = np.array([transition])

            history_d = history_d[-HISTORY_MAX_SIZE:]

            # Calculate rewards for random sample of transitions from history
            history_size = len(history_d)
            sample_indices = np.random.choice(range(history_size), HISTORY_RAND_SAMPLE_SIZE)

            train_states = np.array([history_d[i][0] for i in sample_indices])
            train_y = sess.run(output, feed_dict={x_: train_states})

            # next_states = np.array([history_d[i][3] for i in sample_indices])
            # next_predictions = sess.run(output, feed_dict={x_: next_states})
            for i, sample_index in enumerate(sample_indices):
                # y[i][action] = reward[i] + gamma*q(i+1)
                train_y[i][history_d[sample_index][1]] = history_d[sample_index][2]

                if history_d[sample_index][3] is not None:
                    next_prediction = sess.run(output, feed_dict={x_: [history_d[sample_index][3]]})[0]
                    train_y[i][history_d[sample_index][1]] += history_d[sample_index][2] + REWARD_GAMMA*max(next_prediction)

            # DEBUG INFO:
            # print('predictions: ', predictions)
            print('y: ', train_y)
            # print('action: ', action)
            # print('reward: ', reward)
            # print('Cost: ', sess.run(cost, feed_dict={x_: state_history, y_: y}))

            # Perform train step
            sess.run(train, feed_dict={x_: train_states, y_: train_y})

            # Add summary values
            if _ % 5 == 0:
                summary = sess.run(merged_summary, feed_dict={x_: train_states, y_: train_y})
                summary_writer.add_summary(summary, global_step=_)

            if done:
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="score", simple_value=score),
                ])
                summary_writer.add_summary(summary, global_step=_)

            # Save variables
            if _ % 1000 == 0:
                saver.save(sess, SAVE_DIR + 'saves/save.chkp')

        env.render(mode='human')

        if done:
            env.reset()
            score = 0

run()
