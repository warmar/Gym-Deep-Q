import time
import os
import gym
import gym_ple
import tensorflow as tf
import numpy as np
from collections import deque

GYM_ENV = 'FlappyBird-v0'
SAVE_DIR = './flappy/'
RESUME_SUB_DIR = None  # Set to index of subdirectory e.g. '0/'
MAX_FPS = None
SAVE_CHECKPOINT_STEP_NUM = 1000
SCALED_IMAGE_SIZE = 80
NUM_POSSIBLE_ACTIONS = 2
RANDOM_ACTION_START_RATE = 0.1
RANDOM_ACTION_END_RATE = 0.001
HISTORY_MAX_SIZE = 10000
HISTORY_RAND_SAMPLE_SIZE = 50
REWARD_GAMMA = 0.9
TOTAL_STEPS = 5000000
ACTIVATION_FUNCTION = tf.nn.relu
LEARNING_RATE = 1e-3
DOUBLE_Q = True
UPDATE_TARGET_NETWORK_STEPS = 10000
TAU = 1  # Rate to update target network


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

# Define model functions
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(x, filter_size, stride, out_channels, activation_func, name):
    with tf.name_scope(name):
        in_channels = int(x.shape[3])
        w_name = '%s-weights' % name
        b_name = '%s-bias' % name
        w = weight_variable([filter_size, filter_size, in_channels, out_channels], name=w_name)
        b = bias_variable([out_channels], name=b_name)

        tf.summary.histogram(w_name, w)
        tf.summary.histogram(b_name, b)

        conv = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')
        return activation_func(conv)

def fc_layer(x, out_size, activation_func, name):
    with tf.name_scope(name):
        in_size = int(x.shape[1])
        w_name = '%s-weights' % name
        b_name = '%s-bias' % name
        w = weight_variable([in_size, out_size], name=w_name)
        b = bias_variable([out_size], name=b_name)

        tf.summary.histogram(w_name, w)
        tf.summary.histogram(b_name, b)

        linear = tf.matmul(x, w) + b
        return activation_func(linear)

class GymDeepQ:
    def __init__(self):
        self.sess = tf.Session()
        self._create_environment()
        self._build_model()

    def _create_environment(self):
        self.env = gym.make(GYM_ENV)
        self.env.reset()
        self.image_shape = self.env.render(mode='rgb_array').shape

    def _q_old(self, x):
        conv1 = conv_layer(x, filter_size=8, stride=4, out_channels=32, activation_func=ACTIVATION_FUNCTION, name='conv1')
        conv2 = conv_layer(conv1, filter_size=4, stride=2, out_channels=64, activation_func=ACTIVATION_FUNCTION, name='conv2')
        conv3 = conv_layer(conv2, filter_size=3, stride=1, out_channels=64, activation_func=ACTIVATION_FUNCTION, name='conv3')

        # Flatten image
        num_image_pixels = 1
        for dimension in conv3.shape[1:]:
            num_image_pixels *= int(dimension)
        flattened = tf.reshape(conv3, [-1, num_image_pixels])

        # Fully connected layers
        fc1 = fc_layer(flattened, out_size=512, activation_func=ACTIVATION_FUNCTION, name='fc1')
        fc2 = fc_layer(fc1, out_size=NUM_POSSIBLE_ACTIONS, activation_func=tf.identity, name='fc2')

        return fc2

    def _q(self, x):
        conv1 = conv_layer(x, filter_size=4, stride=4, out_channels=32, activation_func=ACTIVATION_FUNCTION, name='conv1')
        conv2 = conv_layer(conv1, filter_size=4, stride=4, out_channels=64, activation_func=ACTIVATION_FUNCTION, name='conv2')

        last_conv = conv2

        # Flatten image
        num_image_pixels = 1
        for dimension in last_conv.shape[1:]:
            num_image_pixels *= int(dimension)
        flattened = tf.reshape(last_conv, [-1, num_image_pixels])

        # Fully connected layers
        fc1 = fc_layer(flattened, out_size=512, activation_func=ACTIVATION_FUNCTION, name='fc1')
        fc2 = fc_layer(fc1, out_size=NUM_POSSIBLE_ACTIONS, activation_func=tf.identity, name='fc2')

        return fc2

    def _q_target_update_ops(self):
        tf_vars = tf.trainable_variables()
        total_vars = len(tf_vars)
        op_holder = []
        for idx, q_var in enumerate(tf_vars[0:total_vars // 2]):
            q_target_var = tf_vars[idx + (total_vars // 2)]
            op_holder.append(
                q_target_var.assign(
                    (q_var.value() * TAU) + ((1 - TAU) * q_target_var.value())
                )
            )
        return op_holder

    def _update_q_target(self):
        for op in self.update_q_target_ops:
            self.sess.run(op)

    def _build_model(self):
        # Build image processing graph
        self.raw_images = tf.placeholder(tf.float32, shape=[None, *self.image_shape])

        # Scale from [0,255] to [0,1]
        processed_images = tf.multiply(self.raw_images, (1 / 255))
        # Convert to grayscale
        processed_images = tf.image.rgb_to_grayscale(processed_images)
        # Scale down
        processed_images = tf.image.resize_images(processed_images, [SCALED_IMAGE_SIZE, SCALED_IMAGE_SIZE])

        self.processed_images = processed_images

        # Build q function graph
        self.x_ = tf.placeholder(tf.float32, shape=[None, *(processed_images.shape[1:3]), 4])  # processed images placeholder
        self.y_ = tf.placeholder(tf.float32, shape=[None, NUM_POSSIBLE_ACTIONS])  # expected reward placeholder

        # Add input image summary
        tf.summary.image('x_image', self.x_)

        self.q = self._q(self.x_)
        if DOUBLE_Q:
            with tf.name_scope('q_target'):
                self.q_target = self._q(self.x_)

            # Create q_target update ops
            self.update_q_target_ops = self._q_target_update_ops()

        # Define cost function
        cost = tf.reduce_mean(tf.square(self.q - self.y_))
        tf.summary.scalar('cost', cost)

        # Define average Q metric
        avg_q = tf.reduce_mean(self.q)
        tf.summary.scalar('avg_q', avg_q)

        # Define train step
        self.train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

        # Define global_step variable
        self.global_step = tf.Variable(0, trainable=False, name='step')
        self.increment_global_step = tf.assign_add(self.global_step, 1)

        # Initialize all variables in session
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        # Create summary writer
        self.merged_summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(SAVE_DIR + SAVE_SUBDIR, self.sess.graph)

    def restore_model(self):
        self.saver.restore(self.sess, SAVE_DIR + RESUME_SUB_DIR + 'saves/save.chkp')

        history_saves = os.listdir(SAVE_DIR + SAVE_SUBDIR + 'saves/history/')
        for i in range(len(history_saves)):
            self.history_d = np.append(self.history_d, np.load(SAVE_DIR + SAVE_SUBDIR + 'saves/history/history_d%s.npy' % i), axis=0)

    def process_image(self, image):
        processed_image = self.sess.run(self.processed_images, feed_dict={self.raw_images: [image]})[0]
        return processed_image

    def act(self, state, train=False):
        if train:
            # Linearly scale chance of picking a random action
            percent_done = self.get_global_step() / TOTAL_STEPS
            random_chance = RANDOM_ACTION_START_RATE + (RANDOM_ACTION_END_RATE - RANDOM_ACTION_START_RATE) * percent_done

        # While training, occasionally pick random action
        if train and (np.random.random_sample() < random_chance):
            action = np.random.choice(range(NUM_POSSIBLE_ACTIONS))
        else:  # Otherwise pick action with highest predicted reward based on Q function
            predictions = self.sess.run(self.q, feed_dict={self.x_: [state]})[0]
            action = np.argmax(predictions)  # Pick action with biggest predicted reward

        return action

    def do_batch_train_step(self):
        # Train
        history_size = len(self.history_d)

        # Need at least 5 transitions to train
        if history_size < 5:
            return

        # Calculate rewards for random sample of transitions from history
        # Get random sample with enough space for four frames
        sample_start_indices = np.random.choice(range(0, history_size - 4), HISTORY_RAND_SAMPLE_SIZE)

        # Get each value in transitions
        train_states = []
        actions = []
        rewards = []
        next_states = []
        terminal = []
        for sample_start_index in sample_start_indices:
            transitions = np.take(self.history_d, range(sample_start_index, sample_start_index + 5), axis=0)

            curr_frames = deque([transitions[3,0], ], maxlen=4)
            for transition in reversed(transitions[0:3]):
                if transition[3]:  # If terminal, pad frames
                    for _ in range(4 - len(curr_frames)):
                        curr_frames.appendleft(curr_frames[0])
                    break
                curr_frames.appendleft(transition[0])

            next_frames = deque([transitions[4][0], ], maxlen=4)
            for transition in reversed(transitions[1:4]):
                if transition[3]:  # If terminal, pad frames
                    for _ in range(4 - len(next_frames)):
                        next_frames.appendleft(next_frames[0])
                    break
                next_frames.appendleft(transition[0])

            train_states.append(np.dstack(curr_frames))
            next_states.append(np.dstack(next_frames))
            actions.append(transitions[3][1])
            rewards.append(transitions[3][2])
            terminal.append(transitions[3][3])

        # Calculate rewards
        train_y = self.sess.run(self.q, feed_dict={self.x_: train_states})
        next_predictions = self.sess.run(self.q_target if DOUBLE_Q else self.q, feed_dict={self.x_: next_states})
        for i in range(HISTORY_RAND_SAMPLE_SIZE):
            train_y[i][actions[i]] = rewards[i]

            if not terminal[i]:
                train_y[i][actions[i]] += REWARD_GAMMA * max(next_predictions[i])

        # Perform train step
        self.sess.run(self.train_step, feed_dict={self.x_: train_states, self.y_: train_y})

        # Write summary values
        if self.get_global_step() % 25 == 0:
            summary = self.sess.run(self.merged_summary, feed_dict={self.x_: train_states, self.y_: train_y})
            self.summary_writer.add_summary(summary, global_step=self.get_global_step())

        # Save variables and history
        if self.get_global_step() % SAVE_CHECKPOINT_STEP_NUM == 0:
            self.saver.save(self.sess, SAVE_DIR + SAVE_SUBDIR + 'saves/save.chkp')

            # Save history in chunks, deleting the oldest chunk with each save
            history_saves = os.listdir(SAVE_DIR + SAVE_SUBDIR + 'saves/history/')
            i = len(history_saves)
            if len(history_saves) == (HISTORY_MAX_SIZE / SAVE_CHECKPOINT_STEP_NUM):
                os.remove(SAVE_DIR + SAVE_SUBDIR + 'saves/history/history_d0.npy')
                for i in range(1, len(history_saves)):
                    os.rename(SAVE_DIR + SAVE_SUBDIR + 'saves/history/history_d%s.npy' % i,
                              SAVE_DIR + SAVE_SUBDIR + 'saves/history/history_d%s.npy' % (i - 1))
            np.save(SAVE_DIR + SAVE_SUBDIR + 'saves/history/history_d%s.npy' % i, self.history_d[-SAVE_CHECKPOINT_STEP_NUM:])

    def get_global_step(self):
        return self.sess.run(self.global_step)

    def train(self):
        self.run_model(train=True)

    def evaluate(self):
        self.run_model(train=False)

    def run_model(self, train):
        if train:
            self.history_d = np.array([[None, None, None, True],])  # Start with a single terminal transition
            if RESUME_SUB_DIR is not None:
                self.restore_model()

        score = 0
        four_frames = deque(maxlen=4)
        while (not train) or (self.get_global_step() < TOTAL_STEPS):
            start_time = time.time()
            # Get current screen array
            curr_frame = self.process_image(self.env.render(mode='rgb_array'))

            # Create state representation with current frame and previous three frames
            four_frames.append(curr_frame)

            if len(four_frames) == 1:  # If deque is new, pad with current frame
                for _ in range(3):
                    four_frames.append(curr_frame)

            # Convert frames to 4-dimensional image
            curr_state = np.dstack(four_frames)

            # Determine next action
            action = self.act(curr_state, train=train)

            # Perform Action
            observation, reward, done, info = self.env.step(action)
            score += reward

            # Train
            if train:
                # Update history
                transition = [curr_frame, action, reward, done]
                self.history_d = np.append(self.history_d, [transition], axis=0)
                while len(self.history_d) > HISTORY_MAX_SIZE:
                    self.history_d = np.delete(self.history_d, 0, axis=0)

                self.do_batch_train_step()
                if DOUBLE_Q:
                    if self.get_global_step() % UPDATE_TARGET_NETWORK_STEPS == 0:
                        self._update_q_target()

                # Write score summary if done
                if done:
                    summary = tf.Summary(value=[
                        tf.Summary.Value(tag="score", simple_value=score),
                    ])
                    self.summary_writer.add_summary(summary, global_step=self.get_global_step())

                # Increment step
                self.sess.run(self.increment_global_step)

                if self.get_global_step() % 100 == 0:
                    print('Step: ', self.get_global_step())

            if not train:
                self.env.render(mode='human')

            # Reset if done
            if done:
                if not train:
                    print('End - Score: ', score)
                self.env.reset()
                score = 0
                four_frames.clear()

            if MAX_FPS is not None:
                if not train:
                    duration = time.time() - start_time
                    remaining_tick_time = 1 / MAX_FPS - duration
                    if remaining_tick_time > 0:
                        time.sleep(remaining_tick_time)

if __name__ == '__main__':
    learner = GymDeepQ()
    learner.train()
    # learner.evaluate()
