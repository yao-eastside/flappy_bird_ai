import json
import os
import random
import sys
import time
from collections import deque
from datetime import datetime, timedelta

import keras
import numpy as np
import skimage.color
import skimage.exposure
import skimage.transform
from keras import layers, models

from game import wrapped_flappy_bird as game

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
TOTAL_OBSERVATION = 3_200 # timesteps to observe before training
TOTAL_EXPLORE = 30_000 # 3000000. # frames over which to anneal epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
REPLAY_MEMORY = 20_000 # 30000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4


def init_network(mode, observe, epsilon):
    print("Now we build the model structure")

    img_rows, img_cols = 80, 80
    #Convert image into Black and white
    img_channels = 4 # We stack 4 frames

    network = models.Sequential()
    network.add(layers.Conv2D(32, (8, 8), activation='relu', strides=(4, 4), padding='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
    network.add(layers.Conv2D(64, (4, 4), activation='relu', strides=(2, 2), padding='same'))
    network.add(layers.Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    network.add(layers.Flatten())
    network.add(layers.Dense(512, activation='relu'))
    network.add(layers.Dense(2))

    network.compile(loss='mse',optimizer=keras.optimizers.Adam(lr=LEARNING_RATE))
    print("We finish building the model structure")

    if mode == 'test':
        print("testing mode")
        observe = 999999999 # We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        network.load_weights("model.h5")
        print ("Weight load successfully")
    else:
        assert mode == 'train'
        print("training mode")

    return network


def get_init_stack(game_state):
    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t_colored, _, _ = game_state.frame_step(do_nothing)

    x_t = skimage.color.rgb2gray(x_t_colored)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    x_t = x_t / 255.0

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    #print (s_t.shape)

    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4
    return s_t


def get_next_stack(game_state, a_t, s_t):
    #run the selected action and observed next state and reward
    x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

    x_t1 = skimage.color.rgb2gray(x_t1_colored)
    x_t1 = skimage.transform.resize(x_t1,(80,80))
    x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

    x_t1 = x_t1 / 255.0

    x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
    s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

    return s_t1, r_t, terminal


def train_network(queue, network):
    #sample a minibatch to train on
    minibatch = random.sample(queue, BATCH)

    #Now we do the experience replay
    state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
    state_t = np.concatenate(state_t)
    state_t1 = np.concatenate(state_t1)
    targets = network.predict(state_t)
    Q_sa = network.predict(state_t1)
    targets[range(BATCH), action_t] = reward_t + GAMMA*np.max(Q_sa, axis=1)*np.invert(terminal)

    loss = network.train_on_batch(state_t, targets)
    return loss, Q_sa


def logging(t, time0, network, observe, epsilon, action_index, r_t, Q_sa, loss, total_loss):
    # save progress every 10000 iterations
    if t % 1000 == 0:
        save_model(network)

    # print info
    state = "train"
    if t <= observe:
        state = "observe"
    elif t > observe and t <= observe + TOTAL_EXPLORE:
        state = "explore"

    now = datetime.now()
    print(
        now,
        "timespent:", str(timedelta(seconds=time.time() - time0)),
        "TIMESTEP:", t,
        "STATE:", state,
        "EPSILON:", epsilon,
        "ACTION:", action_index,
        "REWARD:", r_t,
        "Q_MAX:" , np.max(Q_sa),
        "Loss:", loss,
        "avgloss:", total_loss/(t+1)
    )


def save_model(network):
    print("Now we save model")
    network.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(network.to_json(), outfile)


def chose_action(network, s_t, a_t, t, epsilon):
    #choose an action epsilon greedy
    if t % FRAME_PER_ACTION == 0:
        if random.random() <= epsilon:
            print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            q = network.predict(s_t)       #input a stack of 4 images, get the prediction
            max_Q = np.argmax(q)
            action_index = max_Q
            a_t[max_Q] = 1
    else:
        assert False


def q_learning(mode):

    observe = TOTAL_OBSERVATION
    epsilon = INITIAL_EPSILON

    # init network
    network = init_network(mode, observe, epsilon)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    queue = deque()

    s_t0 = get_init_stack(game_state)

    t = 0
    time0 = time.time()
    total_loss = 0
    while (True):
        action_index, r_t = 0, 0
        a_t = np.zeros([ACTIONS])
        chose_action(network, s_t0, a_t, t, epsilon)

        # We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > observe:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / TOTAL_EXPLORE

        s_t1, r_t, terminal = get_next_stack(game_state, a_t, s_t0)

        queue.append((s_t0, action_index, r_t, s_t1, terminal))
        if len(queue) > REPLAY_MEMORY:
            queue.popleft()

        if t > observe:
            # only train if done observing
            loss, q_sa = train_network(queue, network)
        else:
            loss, q_sa = 0, 0

        total_loss += loss
        s_t0, t = s_t1, t + 1

        logging(t, time0, network, observe, epsilon, action_index, r_t, q_sa, loss, total_loss)

    print("Episode finished!")
    print("************************")


if __name__ == '__main__':
    TOTAL_OBSERVATION = 32
    q_learning('train')
