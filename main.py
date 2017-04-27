import tensorflow as tf
import numpy as np
from dataset_generator.html_renderer import HTMLGame, HTMLRenderer
input = tf.placeholder(shape=[None, 480*640*3], dtype=tf.float32)
# input_shaped = tf.reshape(input, [1, -1])
W = tf.Variable(tf.random_normal([640*480*3, 6]), name="weights")

Qout = tf.matmul(input, W)
predict = tf.argmax(Qout, 1)

nextQ = tf.placeholder(shape=[1, 6], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

y = .99
e = 0.5
num_episodes = 10000

renderer = HTMLRenderer()

env = HTMLGame('dataset_generator/test_render.png', renderer)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    found_count = 0
    for i in range(num_episodes):
        state = env.reset()
        rAll = 0
        d = False
        j = 0
        # The Q-Network
        while j < 7:
            j += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            a, allQ = sess.run([predict, Qout], feed_dict={input: state})
            if np.random.rand(1) < e:
                a[0] = env.action_sample()
            # Get new state and reward from environment
            next_state, r, d, = env.step(a[0])
            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout, feed_dict={input: next_state})
            # Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, a[0]] = r + y * maxQ1
            # Train our network using target and predicted Q values
            _, W1 = sess.run([updateModel, W], feed_dict={input: state, nextQ: targetQ})
            rAll += r
            state = next_state
            if d:
                # Reduce chance of random action as we train the model.
                e = 1. / ((i / 50) + 10)
                found_count += 1
                break
        if i%10 == 0:
            print('Episode: ', i)
            print('Total reward: ', rAll)
            print('Founded solution: ', found_count)
