import pandas as pd
import numpy as np 
import tensorflow as tf
import asyncio

from keras.models import clone_model


def screenout_body_from_edible(edible, body):
    if not len(body) or len(edible):
        return edible
    
    # (2, n_food, 1) x (2, 1, n_body) -> (2, n_food, n_body)
    b = np.broadcast(edible.T[:, :, None] , body.T[:, None, :])
    out = np.empty(b.shape)
    out.flat = [u == v for (u,v) in b]

    # (2, n_food, n_body) -> (n_food, n_body) -> (n_food, )
    overlaps = out.all(0).any(1)
    return edible[~overlaps]

def get_features(obs):

    # if there's a food object n steps away directly in the front, left, right
    def loc_food_directly_n_steps_away(food_loc, n, nothing_value=4):
        assert nothing_value > n
        if not len(food_loc):
            return nothing_value, nothing_value, nothing_value
                        
        within_range = abs(food_loc).sum(1) <= n
        food_loc = food_loc[within_range]

        if not len(food_loc):
            return nothing_value, nothing_value, nothing_value
        
        ## nothing values will be returned for there's nothing
        food_loc = np.concatenate([food_loc, [[0, -nothing_value], [nothing_value, 0], [-nothing_value, 0]]], axis=0)

        midlineV = food_loc[:, 0] == 0
        midlineH = food_loc[:, 1] == 0

        right = abs(food_loc[(midlineH & (food_loc[:, 0] > 0)), 0]).min()
        left = abs(food_loc[(midlineH & (food_loc[:, 0] < 0)), 0]).min()
        front = abs(food_loc[(midlineV & (food_loc[:, 1] < 0)), 1]).min()

        return left, right, front
    

    food = screenout_body_from_edible(obs['edible'], obs['body'])
    food_L, food_R, food_F = loc_food_directly_n_steps_away(food, n = 10, nothing_value=15 )
    #farfood_L, farfood_R, farfood_F = loc_food_directly_n_steps_away(food, n = 7, nothing_value=8 )


    # if there's a enemy head 2 steps away (including diagonal) in the front, left, right, behind
    def loc_n_step_away(loc, n, nothing_value=4):
        assert nothing_value > n
        if not len(loc):
            return nothing_value, nothing_value, nothing_value, nothing_value
        
        # x + y smaller than n
        within_range = abs(loc).sum(1) <= n
        loc = loc[within_range]
        if not len(loc):
            return nothing_value, nothing_value, nothing_value, nothing_value
        
        # put the nothing values to return
        loc = np.concatenate([loc, [[-nothing_value, -nothing_value], [nothing_value, nothing_value], [-nothing_value, nothing_value],[nothing_value, -nothing_value]]], axis=0)

        # if diagonal: 
        right = abs(loc[(loc[:, 0] > 0), 0]).min()
        left = abs(loc[(loc[:, 0] < 0), 0]).min()
        front = abs(loc[(loc[:, 1] < 0), 1]).min()
        behind = abs(loc[(loc[:, 1] > 0), 1]).min()
        return right, left, front, behind
    
    enemy_R, enemy_L, enemy_F, enemy_B = loc_n_step_away(obs['inedible'], n = 10, nothing_value=15)
    body_R, body_L, body_F, _ = loc_n_step_away(obs['body'], n = 10, nothing_value=15)

    return np.array([food_L, food_R, food_F, body_R, body_L, body_F])

FEATURE_LABELS = ['food_L', 'food_R', 'food_F', 'body_R', 'body_L', 'body_F']




def create_model():
    x = tf.keras.layers.Input(6)
    z = tf.keras.layers.Dense(4)(x)
    return tf.keras.Model(x, z)

class Model:
    def __init__(self, save_name='temp', to_infer=False, to_record=True, model=None):
        """
        fields: 
            action
            [states]
            reward
            obs: objs
            id
        """

        self.save_name=save_name
        self.save_freq = 2000

        if not model is None:
            self.set_model(model)
        else: 
            self.model = create_model()
            self.target_model = create_model()

        self.loss_function = tf.keras.losses.Huber()
        self.optimiser = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        self.update_target_model()

        self.training_step = 0
        self.to_infer = to_infer
        self.to_record = to_record

        self.params = dict(
            target_model_update_freq = 200, 
            gamma = 0.7
        )


        # {agent_id : {actions:list[int], rewards:list[int], states:list[np.ndarray], meta:list[dict]}}
        self.replay_buffer = {}

    def load(self, path):
        model = tf.keras.saving.load_model(path)
        self.set_model(model)

    def set_model(self, model):
        
        self.model = clone_model(model)
        self.model.set_weights(model.get_weights())

        self.target_model = clone_model(model)
        self.target_model.set_weights(model.get_weights())

    def set_to_infer(self, v: bool):
        self.to_infer = v

    def train(self, sample_size=512):
        
        data = self.draw_replay_buffer(sample_size)

        next_q = self.target_model(data['next_state'])

        corrected_q = data['reward'] + tf.reduce_max(next_q, axis=-1) * self.params['gamma']

        action_mask = tf.one_hot(data['action'], depth=next_q.shape[-1])
        with tf.GradientTape() as tape:

            this_q = self.model(data['state'])
            this_q = tf.reduce_sum(tf.multiply(this_q, action_mask), axis=-1)
            loss = self.loss_function(corrected_q, this_q)

        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimiser.apply_gradients(zip(grad, self.model.trainable_variables))

        self.training_step += 1 
        
        if not (self.training_step % self.params['target_model_update_freq']):
            self.update_target_model()
        
        if not (self.training_step % self.save_freq):
            self.model.save(self.save_name+f"_step{self.training_step}.keras")


    async def train_every_n_sec(self, n_sec, sample_size, print_every_n_step=np.inf):
        while True:
            self.train(sample_size)
            await asyncio.sleep(n_sec)
            if not self.training_step % print_every_n_step:
                print(f"step: {self.training_step}", end='       \r')

        

    def draw_replay_buffer(self, sample_size):


        len_by_agent_id = {k:len(v['actions'])-1 for k, v in self.replay_buffer.items()}
        total_sample = sum(len_by_agent_id.values())

        data = dict(state=[], action=[], reward=[], next_state=[])

        for agent_id_, n_sample_ in len_by_agent_id.items():
            sample_to_draw_ = int(sample_size*(n_sample_/total_sample) + 1)

            data_one_agent_ = self.draw_replay_buffer_single_agent(agent_id_, sample_to_draw_)
            
            data['state'].append(data_one_agent_['state'])
            data['action'] += data_one_agent_['action']
            data['reward'] += data_one_agent_['reward']
            data['next_state'].append(data_one_agent_['next_state'])


        indices = np.random.choice(sample_size, sample_size, replace=False)
        data['state'] = np.concatenate(data['state'], axis=0)[indices]
        data['action'] = np.array(data['action'])[indices]
        data['reward'] = np.array(data['reward'])[indices]
        data['next_state'] = np.concatenate(data['next_state'], axis=0)[indices]
        return data

    def draw_replay_buffer_single_agent(self, agent_id, n):
        """
        state (t)
        action (t)
        reward (t+1)
        next state (t+1)
        """
        
        data = self.replay_buffer[agent_id]

        indices = np.random.choice(len(data['actions'])-2, n, replace=False)

        state = [data['states'][idx_] for idx_ in indices]
        action = [data['actions'][idx_] for idx_ in indices]
        reward = [data['rewards'][idx_+1] for idx_ in indices]
        next_state = [data['states'][idx_+1] for idx_ in indices]

        return dict(state=state, action=action, reward=reward, next_state=next_state)
    

    def replay_buffer_df(self):
        df = dict(actions=[], rewards=[], agent_id=[], meta=[])
        for label_ in FEATURE_LABELS:
            df[label_] = []
        
        for agent_id_, data in self.replay_buffer.items():
            df['agent_id'] += [agent_id_]*len(data['actions'])

            df['actions'] += data['actions']
            df['rewards'] += data['rewards']
            df['meta'] += data['meta']


            states = np.stack(data['states'], axis=0)
            for idx, label_ in enumerate(FEATURE_LABELS):
                df[label_] += states[..., idx].tolist()
        return pd.DataFrame(df)


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def append(self, agent_id, states, rewards, actions, meta={}):
        if not agent_id in self.replay_buffer:
            self.replay_buffer[agent_id] = dict(states=[], rewards=[], actions=[], meta=[])
        
        self.replay_buffer[agent_id]['states'].append(states)
        self.replay_buffer[agent_id]['rewards'].append(rewards)
        self.replay_buffer[agent_id]['actions'].append(actions)
        self.replay_buffer[agent_id]['meta'].append(meta)




    def handler(self, data, obs):
        states = get_features(obs)

        if self.to_infer: 
            q_values = self.model(states[None, ...])
            action = int(np.argmax(q_values[0].numpy()))

            assert 'action' not in data['message']
            
        if self.to_record: 
            
            if 'action' in data['message']:
                action = data['message']['action']

            self.append(data['receiver_id'], states=states, rewards=data['message']['rewards'], actions=action)
        
        return action
