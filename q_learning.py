import tensorflow as tf
import numpy as np
from dataset_generator.html_renderer import HTMLGame, HTMLRenderer

VEC_SIZE = 6

input_state = tf.placeholder(shape=[None, 100, 100, 3], dtype=tf.float32)
input_vec = tf.placeholder(shape=[None, VEC_SIZE*6], dtype=tf.float32)

conv1 = tf.layers.conv2d(
      inputs=input_state,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# Convolutional Layer #2 and Pooling Layer #2
conv2 = tf.layers.conv2d(
  inputs=pool1,
  filters=64,
  kernel_size=[3, 3],
  padding="same",
  activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# Dense Layer
pool2_flat = tf.reshape(pool2, [-1, 25 * 25 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

# Logits Layer
conv_state = tf.layers.dense(inputs=dense, units=36)

hidden_num = 100
W = tf.Variable(tf.random_uniform([36 + VEC_SIZE*6, 6]), name="weights")

def model(VEC, STATE, w):
    input = tf.concat((VEC, STATE), axis=1)
    return tf.matmul(input, w)

Qout = model(input_vec, conv_state, W)
Qout_reshaped = tf.reshape(Qout, [6])
predict = tf.argmax(Qout, 1)

nextQ = tf.placeholder(shape=[1, 6], dtype=tf.float32)
reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
# responsible_weight = tf.slice(Qout_reshaped, action_holder, [1])
# loss = -(tf.log(responsible_weight)*reward_holder)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.9)
updateModel = trainer.minimize(loss)

y = .99
e = 0.99
num_episodes = 20000

renderer = HTMLRenderer()

env = HTMLGame('dataset_generator/test_render.png', renderer)

def decode_state(state):
    return [np.argmax(v) for v in np.split(state[0][100*100*3:], VEC_SIZE)]

def expand(v):
    return np.expand_dims(v, axis=0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    found_count = 0
    totalReward = 0
    w_max = 0
    for i in range(num_episodes):
        state, vec = env.reset()
        rAll = 0
        d = False
        j = 0
        # The Q-Network
        while j < 6:
            j += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            a, allQ = sess.run([predict, Qout], feed_dict={input_state: expand(state), input_vec: expand(vec)})
            if np.random.rand(1) < e:
                a[0] = env.action_sample()

            # Get new state and reward from environment
            next_state, vec, r, d, = env.step(a[0])

            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout, feed_dict={input_state: expand(next_state), input_vec: expand(vec)})
            # Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, a[0]] = r + y * maxQ1

            # Train our network using target and predicted Q values
            _, W1 = sess.run([updateModel, W], feed_dict={
                input_state: expand(state),
                nextQ: targetQ,
                action_holder: [a[0]],
                reward_holder: [r],
                input_vec: expand(vec)
            })
            # print('Min W: ', W1.min())
            rAll += r
            state = next_state
            if d:
                # Reduce chance of random action as we train the model.
                e = 1. / ((i / 5000) + 1)
                found_count += 1
                break
        totalReward += rAll
        print('Episode: ', i)
        print('Epsilon: ', e)
        print('Founded solution: ', found_count)
        print('-'*80)
