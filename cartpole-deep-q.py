import os
import gym
import gym_ple
import tensorflow as tf
import numpy as np

GYM_ENV = 'CartPole-v0'
SAVE_DIR = './cartpole/'
LEARNING_RATE = 1e-6
ACTIVATION_FUNCTION = tf.nn.relu
REWARD_GAMMA = 0.95
NUM_POSSIBLE_ACTIONS = 2
PRELIMINARY_RANDOM_ACTIONS = 10000
RANDOM_ACTION_START_RATE = 0.1
RANDOM_ACTION_END_RATE = 0.001
TOTAL_STEPS = 1000000
HISTORY_MAX_SIZE = 50000
HISTORY_RAND_SAMPLE_SIZE = 50
RENDER = True
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
    # Scale from [0,255] to [0,1]
    processed_images = tf.multiply(raw_images, (1 / 255))

    # Convert to grayscale
    processed_images = tf.image.rgb_to_grayscale(processed_images)

    # Scale down
    processed_images = tf.image.resize_images(processed_images, [100, 100])

    # Create computation graph
    x_ = tf.placeholder(tf.float32, shape=[None, *(processed_images.shape[1:3]), 4])
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_POSSIBLE_ACTIONS])

    tf.summary.image('x_image', x_)

    num_channels = int(x_.shape[3])

    # Convolutional Layers
    with tf.name_scope('conv-pool1'):
        w_conv1 = weight_variable([8, 8, num_channels, 32])
        b_conv1 = bias_variable([32])
        tf.summary.histogram('w_conv1', w_conv1)
        tf.summary.histogram('b_conv1', b_conv1)
        conv1 = ACTIVATION_FUNCTION(conv2d(x_, w_conv1, 4) + b_conv1)
        # pool1 = max_pool_2x2(conv1)

    with tf.name_scope('conv-pool2'):
        w_conv2 = weight_variable([4, 4, 32, 64])
        b_conv2 = bias_variable([64])
        tf.summary.histogram('w_conv2', w_conv2)
        tf.summary.histogram('b_conv2', b_conv2)
        conv2 = ACTIVATION_FUNCTION(conv2d(conv1, w_conv2, 2) + b_conv2)
        # pool2 = max_pool_2x2(conv2)

    with tf.name_scope('conv-pool3'):
        w_conv3 = weight_variable([3, 3, 64, 64])
        b_conv3 = bias_variable([64])
        tf.summary.histogram('w_conv3', w_conv3)
        tf.summary.histogram('b_conv3', b_conv3)
        conv3 = ACTIVATION_FUNCTION(conv2d(conv2, w_conv3, 1) + b_conv3)
        # pool3 = max_pool_2x2(conv2)

    # Flatten image
    num_image_pixels = 1
    for dimension in conv3.shape[1:]:
        num_image_pixels *= int(dimension)
    flattened = tf.reshape(conv3, [-1, num_image_pixels])

    # Fully Connected Layers
    with tf.name_scope('fc1'):
        w_fc1 = weight_variable([num_image_pixels, 512])
        b_fc1 = bias_variable([512])
        tf.summary.histogram('w_fc1', w_fc1)
        tf.summary.histogram('b_fc1', b_fc1)
        fc1 = ACTIVATION_FUNCTION(tf.matmul(flattened, w_fc1) + b_fc1)

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

    # Initialize Computation Graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Create Summary Writer
    merged_summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    if not TRAIN:
        saver.restore(sess, SAVE_DIR + 'saves/save.chkp')
    else:
        summary_writer = tf.summary.FileWriter(SAVE_DIR, sess.graph)

    # --- Run Model ---
    history_d = None
    last_four_frames = []

    score = 0
    next_state = env.render(mode='rgb_array')  # Allows us to render the screen only once per step
    next_state = sess.run(processed_images, feed_dict={raw_images: [next_state]})[0]
    for step in range(TOTAL_STEPS):
        # Get current screen array
        curr_state = next_state

        # Create state representation with current frame and previous three frames
        last_four_frames.append(curr_state)

        if len(last_four_frames) > 4:
            del last_four_frames[0]

        # Populate last four frames with first frame if length < 4
        if len(last_four_frames) < 4:
            for _ in range(4 - len(last_four_frames)):
                last_four_frames.insert(0, last_four_frames[0])

        # Convert frames to 4-dimensional image
        curr_state_representation_frames = last_four_frames[:]
        curr_state_representation = np.dstack(curr_state_representation_frames)

        # --- Determine next action ---
        action = None
        # Linearly scale chance of picking a random action
        random_chance = RANDOM_ACTION_START_RATE + (RANDOM_ACTION_END_RATE - RANDOM_ACTION_START_RATE) * (step / TOTAL_STEPS)

        # First populate replay history with random actions, then occationally pick random action
        if ((step < PRELIMINARY_RANDOM_ACTIONS) or
                (np.random.random_sample() < random_chance)):
            action = np.random.choice([0, 1])
        else:
            predictions = sess.run(output, feed_dict={x_: [curr_state_representation]})[0]
            action = 0 if predictions[0] > predictions[1] else 1  # Big action with biggest predicted reward

        # Perform Action
        observation, reward, done, info = env.step(action)
        next_state = env.render(mode='rgb_array')
        next_state = sess.run(processed_images, feed_dict={raw_images: [next_state]})[0]
        score += reward

        # Create next state representation
        next_state_representation_frames = [*last_four_frames[1:], next_state]
        next_state_representation = np.dstack(next_state_representation_frames)

        # Update history
        if not done:
            transition = [curr_state_representation, action, reward, next_state_representation]
        else:
            transition = [curr_state_representation, action, reward, None]

        if history_d is not None:
            history_d = np.vstack((history_d, [transition]))
        else:
            history_d = np.array([transition])

        history_d = history_d[-HISTORY_MAX_SIZE:]

        # Train
        if TRAIN and step >= PRELIMINARY_RANDOM_ACTIONS:
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
            # print('y: ', train_y)
            # print('action: ', action)
            # print('reward: ', reward)
            # print('Cost: ', sess.run(cost, feed_dict={x_: state_history, y_: y}))

            # Perform train step
            sess.run(train, feed_dict={x_: train_states, y_: train_y})

            # Add summary values
            if step % 5 == 0:
                summary = sess.run(merged_summary, feed_dict={x_: train_states, y_: train_y})
                summary_writer.add_summary(summary, global_step=step)

            if done:
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="score", simple_value=score),
                ])
                summary_writer.add_summary(summary, global_step=step)

            # Save variables
            if step % 1000 == 0:
                saver.save(sess, SAVE_DIR + 'saves/save.chkp')

        # Indicators
        if RENDER:
            env.render(mode='human')
        if step % 100 == 0:
            print('Step: ', step)

        # Reset if done
        if done:
            env.reset()
            last_four_frames = []
            score = 0

run()
