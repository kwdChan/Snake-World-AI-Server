from my_types import Action
from .utils import to_sparse
import numpy as np

import tensorflow as tf

VISION_XBOUNDS = (-20, 20)
VISION_YBOUNDS = (-20, 20)
N_ACTION = 4









def model0():
    x_size = VISION_XBOUNDS[1] - VISION_XBOUNDS[0]
    y_size = VISION_YBOUNDS[1] - VISION_YBOUNDS[0]


    x = tf.keras.Input((x_size, y_size, 4))
    z = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')
    z = tf.keras.layers.MaxPooling2D((2, 2))(z)
    z = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(z)
    z = tf.keras.layers.Flatten()(z)
    z = tf.keras.layers.Dense(512, activation='relu')(z)
    z = tf.keras.layers.Dense(256, activation='relu')(z)
    z = tf.keras.layers.Dense(N_ACTION)(z)

    return tf.keras.Model(x, z)

class DeepQAgentOnSegregatedImage:
    def __init__(self, model, optimiser, gamma, train_freq=4, batch_size=32):
        """
        one agent per snake (for training purposes)
        one model shared by different snakes

        """
        self.current_obs = {}
        self.model = model
        self.optimiser = optimiser
        self.gamma = gamma

        # index is t
        self.state_hist = []
        self.action_hist = []
        self.rewards = [None]  # reward after the t = 0

        self.train_freq = train_freq
        self.n_steps = 0
        self.batch_size = batch_size


    def __call__(self, obs,  rewards):

        self.current_obs['no_go'] = no_go =  to_sparse(obs['no_go'], VISION_XBOUNDS, VISION_YBOUNDS)
        self.current_obs['edible'] = edible = to_sparse(obs['edible'], VISION_XBOUNDS, VISION_YBOUNDS)
        self.current_obs['inedible'] = inedible = to_sparse(obs['inedible'], VISION_XBOUNDS, VISION_YBOUNDS)
        self.current_obs['body'] = body = to_sparse(obs['body'], VISION_XBOUNDS, VISION_YBOUNDS)


        ## (x, y, 4)
        state = np.stack([no_go, edible, inedible, body], axis = -1)

        action = Action.RIGHT
        self.state_hist.append(state)
        self.rewards.append(rewards)
        self.action_hist.append(action)

        if not (self.n_steps % self.train_freq):
            self.train()
            
        return action
    
    def half_memory(self):
        pass

    def train(self):
        pass