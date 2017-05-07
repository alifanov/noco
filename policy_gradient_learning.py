import time
import tensorflow as tf
import numpy as np
from dataset_generator.html_renderer import HTMLGame, HTMLRenderer

VEC_SIZE = 6
PG_LR = 0.01
V_LR = 0.1
HIDDEN_LAYER_SIZE = 1000

def policy_gradient():
    with tf.variable_scope("policy"):
        params = tf.Variable(tf.random_uniform([100*100*3 + VEC_SIZE*6, HIDDEN_LAYER_SIZE]), name="weights")
        hidden_params = tf.Variable(tf.zeros([HIDDEN_LAYER_SIZE, 6]), name="hidden_weights")
        state = tf.placeholder(shape=[None, 100*100*3 + VEC_SIZE*6], dtype=tf.float32)
        actions = tf.placeholder(shape=[None, 6], dtype=tf.float32)
        advantages = tf.placeholder("float", [None, 1])

        hidden = tf.matmul(state, params)
        model = tf.matmul(hidden, hidden_params)
        probabilities = tf.nn.softmax(model)
        good_probabilities = tf.reduce_sum(tf.multiply(probabilities, actions), reduction_indices=[1])
        # maximize the log probability
        eligibility = tf.log(good_probabilities) * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(PG_LR).minimize(loss)
        return probabilities, state, actions, advantages, optimizer, loss

def value_gradient():
    with tf.variable_scope("value"):
        state = tf.placeholder(shape=[None, 100*100*3 + VEC_SIZE*6], dtype=tf.float32)
        newvals = tf.placeholder("float",[None,1])
        w1 = tf.get_variable("w1",[100*100*3 + VEC_SIZE*6,HIDDEN_LAYER_SIZE])
        b1 = tf.get_variable("b1",[HIDDEN_LAYER_SIZE])
        h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
        w2 = tf.get_variable("w2",[HIDDEN_LAYER_SIZE,1])
        b2 = tf.get_variable("b2",[1])
        calculated = tf.matmul(h1,w2) + b2
        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(V_LR).minimize(loss)
        return calculated, state, newvals, optimizer, loss

def run_episode(env, policy_grad, value_grad, sess):

    obs_history = []
    actions_history = []
    transitions_history = []
    totalReward = 0
    advantages = []
    update_vals = []

    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer, pl_loss = policy_grad
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad

    # run game
    for game in range(100):
        obs = env.reset()
        for _ in range(6):
            probs = sess.run(pl_calculated, feed_dict={pl_state: obs})
            # print('Probs: ', probs[0])
            action = np.random.choice(6, 1, p=probs[0])
            obs = obs[0]
            obs_history.append(obs)
            actions_history.append(np.identity(6)[action][0])
            old_obs = obs
            # print('Action: ', action[0])
            obs, reward, done = env.step(action[0])
            transitions_history.append((old_obs, action, reward))
            totalReward += reward
            if done:
                print('WIN!!!')
                time.sleep(3)
                break

    # update policy after game
    for index, trans in enumerate(transitions_history):
        obs, action, reward = trans
        future_reward = 0
        future_transitions_count = len(transitions_history) - index
        dec = 1
        for idx in range(future_transitions_count):
            future_reward += transitions_history[idx + index][2] * dec
            dec *= 0.97
        obs_vec = np.expand_dims(obs, axis=0)
        val = sess.run(vl_calculated, feed_dict={vl_state: obs_vec})[0][0]

        advantages.append(future_reward - val)
        update_vals.append(future_reward)

        update_vals_vector = np.expand_dims(update_vals, axis=1)
        ls, _ = sess.run([vl_loss, vl_optimizer], feed_dict={vl_state: obs_history, vl_newvals: update_vals_vector})
        print('Value loss: ', ls)

        advantages_vector = np.expand_dims(advantages, axis=1)
        pl_ls, _ = sess.run([pl_loss, pl_optimizer], feed_dict={
            pl_state: obs_history,
            pl_advantages: advantages_vector,
            pl_actions: actions_history
        })
        print('Policy loss: ', pl_loss)

        return totalReward

num_episodes = 2000

renderer = HTMLRenderer()
env = HTMLGame('dataset_generator/test_render.png', renderer)

policy_grad = policy_gradient()
val_grad = value_gradient()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(num_episodes):
        reward = run_episode(env, policy_grad, val_grad, sess)
        print('Episode: {} | Reward: {}'.format(i+1, reward))