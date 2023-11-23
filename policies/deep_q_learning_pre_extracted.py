from my_types import Action
from .utils import to_sparse
import numpy as np
import pandas as pd
import tensorflow as tf

N_ACTION = 4


def summarise_locations(locs, with_counts=True, with_nearest=True):
    """
    get n left, n right, n in-front, n behind, nearest left, nearest right, nearest in-front nearest behind

    clip the counts at 10 


    negative: top-left
    snake loc: (0,0)

    
    return a list of int
    """
    
    if not len(locs):
        result = []
        if with_counts:
            result += [0,0,0,0]
        if with_nearest: 
            result += [0,0, 0,0, 0,0, 0,0]
        return result

    right = (locs[:, 0] > 0)
    left = (locs[:, 0] < 0)
    centre = (locs[:, 0] == 0)
    front = (centre & (locs[:, 1] < 0))
    behind = (centre & ~front)


    loc_right = locs[right, :]
    loc_left = locs[left, :]
    loc_behind = locs[behind, :]
    loc_front = locs[front, :]

    def get_nearest(locs):
        if not len(locs):
            return [0, 0]
        locs = abs(locs)
        #print(locs)
        argmin = np.argmin(locs[:, 0]+locs[:, 1])
        return locs[argmin,0], locs[argmin,1]
    
    result = []
    if with_counts: 
        
        result += [max(10, len(loc_right)),max(10, len(loc_left)),max(10, len(loc_behind)),max(10, len(loc_front))]

    if with_nearest:
        result += [
            *get_nearest(loc_right), 
            *get_nearest(loc_left), 
            *get_nearest(loc_behind), 
            *get_nearest(loc_front), 
        ]
    
    return result

class DQLModelPreExtracted:
    def __init__(self):
        def create_model():
            x = tf.keras.layers.Input(36)
            z = tf.keras.layers.Dense(50, activation='relu')(x)
            z = tf.keras.layers.Dense(4, activation='linear')(z)
            return tf.keras.Model(x, z)
        
        self.model = create_model()
        self.target_model = create_model()

        # from https://keras.io/examples/rl/deep_q_network_breakout/
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        self.loss_function = tf.keras.losses.Huber()
        self.gamma = 0.7

        self.max_agent_number = 20

        self.max_replay_buffer_size=10000
        self.buffer_clearing_size = 1000
        self.update_target=20

        self.epsilon_random_frames = 10000
        self.epsilon_greedy_frames = 1000000

        self.epsilon = 1
        self.epsilon_min = 0.1  # Minimum epsilon greedy parameter
        self.epsilon_max = 1.0  # Maximum epsilon greedy parameter
        self.epsilon_interval = (
            self.epsilon_max - self.epsilon_min
        )  # Rate at which to reduce chance of random action being taken


        # variables
        self.train_step = 0
        self.agent_list = []

        self.agent_total_steps = 0

    def increment_agent_total_steps(self, agent:"Agent"):
        self.agent_total_steps += 1
        if self.agent_total_steps < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
            agent.make_next_action_random()
        self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
    
    def train(self, n_sample):
        agent_list = self.agent_list

        def draw_samples(agent_list, n_sample):
            total_length = sum([a_.replay_buffer_size for a_ in agent_list])

            s = []
            r = []
            a = []
            ns = []

            for agent_ in agent_list:
                n_sample = int(n_sample*(agent_.replay_buffer_size/total_length))+1
                hist = agent_.draw_history(n_sample)
                s += hist["state"]
                r += hist["reward"]
                a += hist["action"]
                ns += hist["next_state"]

            s, r, a, ns = np.concatenate(s, axis=0), np.array(r), np.array(a), np.concatenate(ns, axis=0)
            indices = np.random.choice(n_sample, n_sample, replace=False)

            return s[indices], r[indices], a[indices], ns[indices]
        
        s, r, a, ns = draw_samples(agent_list, n_sample)
        
        #print(ns.shape)
        q_ns: tf.Tensor = self.target_model(ns) #(sample, action)

        updated_q_s = r + self.gamma*tf.reduce_max(q_ns, axis=1)

        masks = tf.one_hot(a, q_ns.shape[-1])

        s_tensor = tf.convert_to_tensor(s)
        updated_q_s_tensor = tf.convert_to_tensor(updated_q_s)

        with tf.GradientTape() as tape:
            q_s = self.model(s_tensor)

            q_action = tf.reduce_sum(tf.multiply(q_s, masks), axis=1)
            loss = self.loss_function(updated_q_s_tensor, q_action)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_step += 1

        if not(self.train_step % self.update_target):
            self.target_model.set_weights(self.model.get_weights())

    def eval(self):
        r = []
        ts = []
        for a_ in self.agent_list:
            r += a_.reward_hist
            ts += a_.training_step
        
        return pd.DataFrame([r, ts], index=['rewards', 'step']).T

    def create_agent(self, agent_id):
        
        new_agent = type(self).Agent(agent_id, self, self.max_replay_buffer_size, self.buffer_clearing_size)
        self.agent_list.append(new_agent)
        if len(self.agent_list) > self.max_agent_number:
            self.agent_list = self.agent_list[-self.max_agent_number:]
        return new_agent
    
    def get_agent(self, agent_id):
        
        for agent_ in self.agent_list:
            if agent_.agent_id == agent_id:
                return agent_
        return self.create_agent(agent_id)
    

    

    class Agent:
        def __init__(self, agent_id, model:"DQLModelPreExtracted", max_replay_buffer_size, buffer_clearing_size):
            """
            one agent per snake (for training purposes)
            one model shared by different snakes

            model is a DeepQAgentOnPreExtracted.Model object
            """
            self.current_obs = {}
            self.model = model
            self.agent_id = agent_id

            # index is t
            self.state_hist = []
            self.action_hist = []
            self.reward_hist = []  # reward for the previous state/action
            self.training_step = []

            self.max_replay_buffer_size = max_replay_buffer_size
            self.buffer_clearing_size = buffer_clearing_size
            self.training_episode = 0

            self.next_action_random = False

        def make_next_action_random(self):
            self.next_action_random = True
        
        @property
        def replay_buffer_size(self):
            return len(self.state_hist) - 1
            
        @staticmethod
        def get_features(obs):
            return np.array([
                #summarise_locations(obs['no_go'], with_counts=False) +  #8
                summarise_locations(obs['edible']) + #12
                summarise_locations(obs['inedible']) + #12
                summarise_locations(obs['body']) #12
            ] )  #44

        def __call__(self, obs,  rewards):
            
            state = type(self).get_features(obs)  

            self.model.increment_agent_total_steps(self)
            if not self.next_action_random: 
                q_values = self.model.model(state) # (1, n_action)
                action = int(np.argmax(q_values.numpy()[0]))
            else: 
                action = int(np.random.random_integers(0, 3)) # inclusive
                self.next_action_random = False


            if self.replay_buffer_size > self.max_replay_buffer_size:
                self.state_hist = self.state_hist[self.buffer_clearing_size:]
                self.action_hist = self.action_hist[self.buffer_clearing_size:]
                self.reward_hist = self.rewards[self.buffer_clearing_size:]

            self.reward_hist.append(rewards)        
            self.state_hist.append(state)
            self.action_hist.append(action)
            self.training_step.append(self.model.train_step)
            



            return action

        def draw_history(self, size):

            
            if self.replay_buffer_size<3:
                return {
                "state":[],
                "action":[],
                "reward":[],
                "next_state":[],
            }
        
            if (size == 0):
                sample_indices = []
            else: 
                sample_indices = np.random.choice(self.replay_buffer_size, size)

            return {
                "state":[self.state_hist[idx] for idx in sample_indices],
                "action":[self.action_hist[idx] for idx in sample_indices],
                "reward":[self.reward_hist[idx+1] for idx in sample_indices],
                "next_state":[self.state_hist[idx+1] for idx in sample_indices],
            }
        
        
