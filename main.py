import tensorflow as tf
import numpy as np
from dataset_generator.html_renderer import HTMLGame, HTMLRenderer
input = tf.placeholder(shape=[None, 100*100*3 + 3], dtype=tf.float32)
# input_shaped = tf.reshape(input, [1, -1])
W = tf.Variable(tf.zeros([100*100*3 + 3, 4]), name="weights")

Qout = tf.matmul(input, W)
predict = tf.argmax(Qout, 1)

nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.9)
updateModel = trainer.minimize(loss)

negative_memory = []

y = .99
e = 0.5
num_episodes = 600000

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
        while j < 3:
            j += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            a, allQ = sess.run([predict, Qout], feed_dict={input: state})
            state_hash = hash(state.tostring())
            if np.random.rand(1) < e:
                a[0] = env.action_sample()

            memory_item = (state_hash, a[0])
            # print(memory_item)
            if memory_item in negative_memory:
                for action in env.action_samples():
                    memory_item = (state_hash, action)
                    if memory_item not in negative_memory:
                        a[0] = action
                        break
            # Get new state and reward from environment
            next_state, r, d, = env.step(a[0])

            # save in memory
            if r < 0:
                memory_item = (state_hash, a[0])
                if memory_item not in negative_memory:
                    negative_memory.append(memory_item)

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
                e *= e
                found_count += 1
                break
        if i % 50 == 0:
            print('Episode: ', i)
            print('Total reward: ', rAll)
            print('Founded solution: ', found_count)
            print('Negative memory size: ', len(negative_memory))
