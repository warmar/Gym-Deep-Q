import os
import gym
import gym_ple
import tensorflow as tf
import numpy as np

GYM_ENV = 'CartPole-v0'
SAVE_DIR = './cartpole/'
LEARNING_RATE = 1e-5
ACTIVATION_FUNCTION = tf.nn.relu
REWARD_GAMMA = 0.95
NUM_POSSIBLE_ACTIONS = 2
PRELIMINARY_RANDOM_ACTIONS = 10000
RANDOM_ACTION_START_RATE = 0.1
RANDOM_ACTION_END_RATE = 0.001
TOTAL_STEPS = 5000000
HISTORY_MAX_SIZE = 50000
HISTORY_RAND_SAMPLE_SIZE = 50
SAVE_CHECKPOINT_STEP_NUM = 1000
RENDER = False
TRAIN = True  # True to train, False to run trained model
RESUME_SUB_DIR = None  # Set to index of subdirectory e.g. '0/'

# Determine index for current run
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
runs = os.listdir(SAVE_DIR)
i = 0
while str(i) in runs:
    i += 1
SAVE_SUBDIR = '%d/' % i

# If resuming, use RESUME_DIR
if RESUME_SUB_DIR is not None:
    SAVE_SUBDIR = RESUME_SUB_DIR

# Create necessary directories
if not os.path.exists(SAVE_DIR + SAVE_SUBDIR):
    os.mkdir(SAVE_DIR + SAVE_SUBDIR)
    os.mkdir(SAVE_DIR + SAVE_SUBDIR + 'saves/')
    os.mkdir(SAVE_DIR + SAVE_SUBDIR + 'saves/history/')

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

    # Define average Q metric
    avg_q = tf.reduce_mean(output)
    tf.summary.scalar('avg_q', avg_q)

    # Define train step
    train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    # Define global_step variable
    global_step = tf.Variable(0, trainable=False, name='step')
    increment_global_step = tf.assign_add(global_step, 1)

    # Initialize Computation Graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Create saver and restore previous run if applicable
    saver = tf.train.Saver()
    if RESUME_SUB_DIR is not None:
        saver.restore(sess, SAVE_DIR + RESUME_SUB_DIR + 'saves/save.chkp')
    if TRAIN:
        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(SAVE_DIR + SAVE_SUBDIR, sess.graph)

    # Run Model
    if TRAIN and RESUME_SUB_DIR is not None:
        history_saves = os.listdir(SAVE_DIR + SAVE_SUBDIR + 'saves/history/')
        history_d = np.load(SAVE_DIR + SAVE_SUBDIR + 'saves/history/history_d0.npy')
        for i in range(1, len(history_saves)):
             history_d = np.append(history_d, np.load(SAVE_DIR + SAVE_SUBDIR + 'saves/history/history_d%s.npy' % i), axis=0)
    else:
        history_d = np.array([[None, None, None, True]])  # Start with a single terminal transition

    score = 0
    next_state = env.render(mode='rgb_array')  # Allows us to render the screen only once per step
    next_state = sess.run(processed_images, feed_dict={raw_images: [next_state]})[0]
    while global_step.eval(sess) < TOTAL_STEPS:
        # Get current screen array
        curr_state = next_state

        # Create state representation with current frame and previous three frames
        curr_four_frames = [curr_state]
        for i in [-1, -2, -3]:
            transition_i = history_d[i]
            if transition_i[3]:  # Previous state was terminal; do not use; pad with last input frame
                for _ in range(4-len(curr_four_frames)):
                    curr_four_frames.insert(0, curr_four_frames[0])
                break
            transition_state = transition_i[0]
            curr_four_frames.insert(0, transition_state)

        # Convert frames to 4-dimensional image
        curr_state_representation = np.dstack(curr_four_frames)

        # --- Determine next action ---
        # Linearly scale chance of picking a random action
        random_chance = RANDOM_ACTION_START_RATE + (RANDOM_ACTION_END_RATE - RANDOM_ACTION_START_RATE) * (global_step.eval(sess) / TOTAL_STEPS)

        # While training, pick random action if still in preliminary building replay memory stage, then occasionally after
        if TRAIN and \
                ((global_step.eval(sess) < PRELIMINARY_RANDOM_ACTIONS) or
                (np.random.random_sample() < random_chance)):
            action = np.random.choice(range(NUM_POSSIBLE_ACTIONS))
        else:  # Otherwise pick action with highest predicted reward based on Q function
            predictions = sess.run(output, feed_dict={x_: [curr_state_representation]})[0]
            action = np.argmax(predictions)  # Pick action with biggest predicted reward

        # Perform Action
        observation, reward, done, info = env.step(action)
        score += reward

        # Pre-process next frame
        next_state = env.render(mode='rgb_array')
        next_state = sess.run(processed_images, feed_dict={raw_images: [next_state]})[0]

        # Update history
        transition = [curr_state, action, reward, done]

        history_d = np.append(history_d, [transition], axis=0)
        if len(history_d) > HISTORY_MAX_SIZE:
            history_d = np.delete(history_d, 0, axis=0)

        # Train
        if TRAIN:
            # Train only when replay memory is filled with enough random actions
            if global_step.eval(sess) >= PRELIMINARY_RANDOM_ACTIONS:
                # Calculate rewards for random sample of transitions from history
                # Get random sample
                history_size = len(history_d)
                sample_indices = np.random.choice(range(3, history_size-1), HISTORY_RAND_SAMPLE_SIZE)

                # Get each value in transitions
                train_states = []
                actions = []
                rewards = []
                next_states = []
                terminal = []
                for sample_index in sample_indices:
                    prev_transitions = np.take(history_d, [sample_index-1, sample_index-2, sample_index-3], axis=0)
                    curr_transition = history_d[sample_index]
                    next_transition = history_d[sample_index+1]

                    curr_frames = [curr_transition[0]]
                    for prev_transition in prev_transitions:
                        if prev_transition[3]:  # If terminal, pad frames
                            for _ in range(4-len(curr_frames)):
                                curr_frames.insert(0, curr_frames[0])
                            break
                        curr_frames.insert(0, prev_transition[0])

                    next_frames = [curr_transition[0], next_transition[0]]
                    for prev_transition in prev_transitions[:2]:
                        if prev_transition[3]:  # If terminal, pad frames
                            for _ in range(4 - len(next_frames)):
                                next_frames.insert(0, next_frames[0])
                            break
                        next_frames.insert(0, prev_transition[0])

                    train_states.append(np.dstack(curr_frames))
                    next_states.append(np.dstack(next_frames))
                    actions.append(curr_transition[1])
                    rewards.append(curr_transition[2])
                    terminal.append(curr_transition[3])

                # Calculate rewards
                train_y = sess.run(output, feed_dict={x_: train_states})
                next_predictions = sess.run(output, feed_dict={x_: next_states})
                for i in range(HISTORY_RAND_SAMPLE_SIZE):
                    train_y[i][actions[i]] = rewards[i]

                    if not terminal[i]:
                        train_y[i][actions[i]] += REWARD_GAMMA*max(next_predictions[i])

                # DEBUG INFO:
                # print('predictions: ', predictions)
                # print('y: ', train_y)
                # print('action: ', action)
                # print('reward: ', reward)
                # print('Cost: ', sess.run(cost, feed_dict={x_: state_history, y_: y}))

                # Perform train step
                sess.run(train, feed_dict={x_: train_states, y_: train_y})

                # Add summary values
                if global_step.eval(sess) % 25 == 0:
                    summary = sess.run(merged_summary, feed_dict={x_: train_states, y_: train_y})
                    summary_writer.add_summary(summary, global_step=global_step.eval(sess))

                if done:
                    summary = tf.Summary(value=[
                        tf.Summary.Value(tag="score", simple_value=score),
                    ])
                    summary_writer.add_summary(summary, global_step=global_step.eval(sess))

            # Save variables and history
            if global_step.eval(sess) % SAVE_CHECKPOINT_STEP_NUM == 0:
                saver.save(sess, SAVE_DIR + SAVE_SUBDIR + 'saves/save.chkp')

                # Save history in chunks, deleting the oldest chunk with each save
                history_saves = os.listdir(SAVE_DIR + SAVE_SUBDIR + 'saves/history/')
                i = len(history_saves)
                if len(history_saves) == (HISTORY_MAX_SIZE/SAVE_CHECKPOINT_STEP_NUM):
                    os.remove(SAVE_DIR + SAVE_SUBDIR + 'saves/history/history_d0.npy')
                    for i in range(1, len(history_saves)):
                        os.rename(SAVE_DIR + SAVE_SUBDIR + 'saves/history/history_d%s.npy' % i, SAVE_DIR + SAVE_SUBDIR + 'saves/history/history_d%s.npy' % (i-1))
                np.save(SAVE_DIR + SAVE_SUBDIR + 'saves/history/history_d%s.npy' % i, history_d[-SAVE_CHECKPOINT_STEP_NUM:])

            # Increment Step
            sess.run(increment_global_step)

            if global_step.eval(sess) % 100 == 0:
                print('Step: ', global_step.eval(sess))

        if RENDER:
            env.render(mode='human')

        # Reset if done
        if done:
            if not TRAIN:
                print('End - Score: ', score)
            env.reset()
            score = 0


run()
