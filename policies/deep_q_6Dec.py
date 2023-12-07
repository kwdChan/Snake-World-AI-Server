import pandas as pd
import numpy as np 
import tensorflow as tf
import asyncio

from typing import List

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


def is_coords_in_region(coords, xlim, ylim):
    """
    coords: (n, 2)

    xlim/ylim are boundary exclusive
    """
    if not len(coords):
        return np.array([])

    in_region = np.apply_along_axis(lambda s: ( (s[0]>xlim[0]) & (s[0]<xlim[1]) & (s[1]>ylim[0]) & (s[1]<ylim[1]) ), coords, axis=-1)
    return in_region
    
def get_nearest_coord(coords, nothing_value):
    """
    coords: (n, 2)

    xlim/ylim are boundry exclusive
    """
    if not len(coords):
        return nothing_value
    
    index = np.argmin(abs(coords).sum(1))
    coord = coords[index] 
    return coord[0], coord[1]

def four_quadrants_bool(coords):

    if not len(coords):
        raise ValueError("zero length coords")

    in_front_bool = coords[..., 1] < 1
    in_left_bool = coords[..., 0] < 0
    in_right_bool = coords[..., 0] > 0
    in_middle_bool = coords[..., 0] == 0

    return in_front_bool, in_left_bool, in_right_bool, in_middle_bool



def is_near(coords, dist):
    return (abs(coords).sum(1) < dist)


NOTHING_FAR = (200, 200)
NOTHING_NEAR = (20, 20)

def get_features(obs):
    """

    nearest food (front) x, y
    
    counts food (near front left)
    counts food (near front right)

    counts food (full front left)
    counts food (full front right)
    counts food (full back left)
    counts food (full back right)    

    nearest enemy (full range) x, y
    counts enemy (near front left)
    counts enemy (near front right)
    counts enemy (near back left)
    counts enemy (near back right)

    nearest body (front-middle) y
    nearest body (front left) x, y
    nearest body (front right) x, y

    counts body (full front left) 
    counts body (full front right) 
    counts body (full back middle) 
    counts body (full back left) 
    counts body (full back right) 


    dist world boundary (left, right, front, back) l, r, f, b

    obs['edible']
    obs['inedible']
    obs['body']
    obs['no_go']
    """


    # food

    def get_food_features(feature_dict):
        """
        nearest food (front) x, y
    
        counts food (near front left)
        counts food (near front right)

        counts food (full front left)
        counts food (full front right)
        counts food (full back left)
        counts food (full back right)    

        """
        foods = screenout_body_from_edible(obs['edible'], obs['body'])

        if len(foods):
            foods_front_bool, foods_left_bool, foods_right_bool, foods_middle_bool = four_quadrants_bool(foods)

            # nearest food (front) x, y
            feature_dict['nearest_food_x'], feature_dict['nearest_food_y'] = get_nearest_coord(foods[foods_front_bool], nothing_value=(NOTHING_FAR[0], -NOTHING_FAR[1]))

            # counts food (full front left)
            feature_dict['count_food_full_front_left'] = (foods_front_bool & foods_left_bool).sum()
            feature_dict['count_food_full_front_right'] = (foods_front_bool & foods_right_bool).sum()
            feature_dict['count_food_full_back_left'] = ((~foods_front_bool) & foods_left_bool).sum()
            feature_dict['count_food_full_back_right'] = ((~foods_front_bool) & foods_right_bool).sum()
            feature_dict['count_food_near_front_left'] = is_near(foods[foods_front_bool & foods_left_bool], dist=20).sum()
            feature_dict['count_food_near_front_right'] = is_near(foods[foods_front_bool & foods_right_bool], dist=20).sum()

        else: 
            feature_dict['nearest_food_x'], feature_dict['nearest_food_y'] = (NOTHING_FAR[0], -NOTHING_FAR[1])
            feature_dict['count_food_full_front_left'] = 0
            feature_dict['count_food_full_front_right'] = 0
            feature_dict['count_food_full_back_left'] = 0
            feature_dict['count_food_full_back_right'] = 0
            feature_dict['count_food_near_front_left'] = 0
            feature_dict['count_food_near_front_right'] = 0


    def get_enemy_features(feature_dict):
        """
        nearest enemy (full range) x, y
        counts enemy (near front left)
        counts enemy (near front right)
        counts enemy (near back left)
        counts enemy (near back right)
        """
        enemies = obs['inedible']
        

        if len(enemies):
            enemies = enemies[~((enemies[..., 0] == 0) & (enemies[..., 1] == 0))]
   
            front_bool, left_bool, right_bool, middle_bool = four_quadrants_bool(enemies)

            feature_dict['nearest enemy x'], feature_dict['nearest enemy y'] = get_nearest_coord(enemies, nothing_value=NOTHING_FAR)

            feature_dict['count enemy near front left'] = is_near(enemies[(front_bool & left_bool)], dist=20).sum()
            feature_dict['count enemy near front right'] = is_near(enemies[(front_bool & right_bool)], dist=20).sum()
            feature_dict['count enemy near back left'] = is_near(enemies[(~front_bool & left_bool)], dist=20).sum()
            feature_dict['count enemy near back right'] = is_near(enemies[(~front_bool & right_bool)], dist=20).sum()
        
        else:
            feature_dict['nearest enemy x'], feature_dict['nearest enemy y'] = NOTHING_FAR # TODO: this value is bottom left

            feature_dict['count enemy near front left'] = 0
            feature_dict['count enemy near front right'] = 0
            feature_dict['count enemy near back left'] = 0
            feature_dict['count enemy near back right'] = 0
        
        

    def get_body_features(feature_dict):
        """
        nearest body (front-middle) y
        nearest body (front left) x, y
        nearest body (front right) x, y

        counts body (full front left) 
        counts body (full front right) 
        counts body (full back middle) 
        counts body (full back left) 
        counts body (full back right) 
        """
        bodies = obs['body']

        if len(bodies):
            front_bool, left_bool, right_bool, middle_bool = four_quadrants_bool(bodies)

            _, feature_dict['nearest body front middle y'] = get_nearest_coord(bodies[front_bool & middle_bool], nothing_value=(0, -NOTHING_FAR[1]))
            feature_dict['nearest body front left x'], feature_dict['nearest body front left y'] = get_nearest_coord(bodies[front_bool & left_bool], nothing_value=(-NOTHING_FAR[0], -NOTHING_FAR[1])) 
            feature_dict['nearest body front right x'], feature_dict['nearest body front right y'] = get_nearest_coord(bodies[front_bool & right_bool], nothing_value=(NOTHING_FAR[0], -NOTHING_FAR[1]))

            feature_dict['count body full front right'] = (front_bool & right_bool).sum()
            feature_dict['count body full front left']  = (front_bool & left_bool).sum()

            feature_dict['count body full back middle'] = ((~front_bool) & middle_bool).sum()

            feature_dict['count body full back left'] = ((~front_bool) & left_bool).sum()
            feature_dict['count body full back right'] = ((~front_bool) & right_bool).sum()


        else:
            _, feature_dict['nearest body front middle y'] = 0, -NOTHING_FAR[1]
            feature_dict['nearest body front left x'], feature_dict['nearest body front left y'] = -NOTHING_FAR[0], -NOTHING_FAR[1]
            feature_dict['nearest body front right x'], feature_dict['nearest body front right y'] = NOTHING_FAR[0], -NOTHING_FAR[1]

            feature_dict['count body full front right'] = 0
            feature_dict['count body full front left']  = 0

            feature_dict['count body full back middle'] = 0

            feature_dict['count body full back left'] = 0
            feature_dict['count body full back right'] = 0



    def get_world_boundary(feature_dict):
        # when the snake is dead, obs['no_go'] has no length. I Don't Know Why

        feature_dict['boundary_x_min'] = obs['no_go'][..., 0].min() if len(obs['no_go']) else 0
        feature_dict['boundary_x_max'] = obs['no_go'][..., 0].max() if len(obs['no_go']) else 0
        feature_dict['boundary_y_min'] = obs['no_go'][..., 1].min() if len(obs['no_go']) else 0
        feature_dict['boundary_y_max'] = obs['no_go'][..., 1].max() if len(obs['no_go']) else 0
        
    # for potential hazard: the order of the dictionary 

    feature_dict = {
        "nearest_food_x": None, 
        "nearest_food_y": None, 
        "count_food_full_front_left": None, 
        "count_food_full_front_right": None, 
        "count_food_full_back_left": None, 
        "count_food_full_back_right": None, 
        "count_food_near_front_left": None, 
        "count_food_near_front_right": None, 
        "nearest enemy x": None, 
        "nearest enemy y": None, 
        "count enemy near front left": None, 
        "count enemy near front right": None, 
        "count enemy near back left": None, 
        "count enemy near back right": None, 

        "nearest body front middle y": None, 
        "nearest body front left x": None, 
        "nearest body front left y": None, 
        "nearest body front right x": None, 
        "nearest body front right y": None, 
        "count body full front right": None, 
        "count body full front left": None, 
        "count body full back middle": None, 
        "count body full back left": None, 
        "count body full back right": None, 

        'boundary_x_min': None,
        'boundary_x_max': None,
        'boundary_y_min': None,
        'boundary_y_max': None,

    }

    get_food_features(feature_dict)
    get_enemy_features(feature_dict)
    get_body_features(feature_dict)
    get_world_boundary(feature_dict)



    return feature_dict



def create_model():
    x = tf.keras.layers.Input(28)/10
    z = tf.keras.layers.Dense(100, activation='elu')(x)
    z = tf.keras.layers.Dense(20, activation='elu')(z)
    z = tf.keras.layers.Dense(4)(z)
    return tf.keras.Model(x, z)

class Model:
    def __init__(self, save_name='./data/model_6Dec', save_freq=2000, model: tf.keras.Model=None, to_record=True):
        """
        """

        self.save_name = save_name
        self.save_freq = save_freq
        self.to_record = to_record


        # create the model
        if not model is None:
            self.set_model(model)
        else: 
            self.model = create_model()
            self.target_model = create_model()
        self.update_target_model()

        # training parameter
        self.loss_function = tf.keras.losses.Huber()
        self.optimiser = tf.keras.optimizers.Adam(learning_rate=0.000025, clipnorm=1.0)


        self.monitoring_params = dict(
            freq = 1000, 
            duration = 20
        )

        self.training_params = dict(
            target_model_update_freq = 50, 
            gamma = 0.8, 
            
            # the the length of the replay buffer is above this number, all memory is cleared and training will resume only after it has enough data (1.5 * batch size)
            max_replay_buffer_length = 10000000, 

            # epsilon decreases from max to min over time after `epsilon_random_step` step
            # epsilon is the probablity of doing a random action 
            epsilon_min = 0.1, 
            epsilon_max = 1.0, 
            epsilon_greedy_step = 100000, 
            epsilon_random_step = 10000
        )
        self.training_params['epsilon_interval'] = self.training_params['epsilon_max'] - self.training_params['epsilon_min']


        # variables to be updated by the object
        # {agent_id : {actions:list[int], rewards:list[int], states:list[np.ndarray], meta:list[dict]}}
        self.replay_buffer = {}
        self.training_step = 0
        self.replay_buffer_length = 0
        self.total_step = 0
        self.epsilon = self.training_params['epsilon_max']
        self.feature_labels = None
        self.error_obs = []
        self.monitoring_checkpoints = []


    def load(self, path):
        model = tf.keras.saving.load_model(path)
        self.set_model(model)

    def set_model(self, model: tf.keras.Model):
        
        self.model = clone_model(model)
        self.model.set_weights(model.get_weights())

        self.target_model = clone_model(model)
        self.target_model.set_weights(model.get_weights())

    def set_to_infer(self, v: bool):
        self.to_infer = v

    def train(self, sample_size=512):

        if self.replay_buffer_length < sample_size*1.5:
            return 
        
        data = self.draw_replay_buffer(sample_size)

        next_q = self.target_model(data['next_state'])

        corrected_q = data['reward'] + tf.reduce_max(next_q, axis=-1) * self.training_params['gamma']

        action_mask = tf.one_hot(data['action'], depth=next_q.shape[-1])
        with tf.GradientTape() as tape:

            this_q = self.model(data['state'])
            this_q = tf.reduce_sum(tf.multiply(this_q, action_mask), axis=-1)
            loss = self.loss_function(corrected_q, this_q)

        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimiser.apply_gradients(zip(grad, self.model.trainable_variables))

        self.training_step += 1 
        
        if not (self.training_step % self.training_params['target_model_update_freq']):
            self.update_target_model()
        
        if not (self.training_step % self.save_freq):
            self.model.save(self.save_name+f"_step{self.training_step}.keras")

        if (self.training_step  % self.monitoring_params['freq']) < self.monitoring_params['duration']:
            self.monitoring_checkpoints.append(self.model.get_weights())

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
            #print(sample_to_draw_)
            data_one_agent_ = self.draw_replay_buffer_single_agent(agent_id_, sample_to_draw_)
            
            if len(data_one_agent_['state']): 

                data['state'] += data_one_agent_['state']
                data['action'] += data_one_agent_['action']
                data['reward'] += data_one_agent_['reward']
                data['next_state'] += data_one_agent_['next_state']


        indices = np.random.choice(sample_size, sample_size, replace=False)
        data['state'] = np.stack(data['state'], axis=0)[indices]
        data['action'] = np.array(data['action'])[indices]
        data['reward'] = np.array(data['reward'])[indices]
        data['next_state'] = np.stack(data['next_state'], axis=0)[indices]
        return data

    def draw_replay_buffer_single_agent(self, agent_id, n):
        """
        state (t)
        action (t)
        reward (t+1)
        next state (t+1)
        """
        
        data = self.replay_buffer[agent_id]

        if (n > len(data['actions'])-2) or ((len(data['actions'])-2) < 1):
            return dict(state=[], action=[], reward=[], next_state=[])

        indices = np.random.choice(len(data['actions'])-2, n, replace=False)

        state: List[np.ndarray] = [data['states'][idx_] for idx_ in indices]
        action = [data['actions'][idx_] for idx_ in indices]
        reward = [data['rewards'][idx_+1] for idx_ in indices]
        next_state: List[np.ndarray] = [data['states'][idx_+1] for idx_ in indices]

        return dict(state=state, action=action, reward=reward, next_state=next_state)
    

    def replay_buffer_df(self):
        df = dict(actions=[], rewards=[], agent_id=[], meta=[])
        for label_ in self.feature_labels:
            df[label_] = []
        
        for agent_id_, data in self.replay_buffer.items():
            df['agent_id'] += [agent_id_]*len(data['actions'])

            df['actions'] += data['actions']
            df['rewards'] += data['rewards']
            df['meta'] += data['meta']


            states = np.stack(data['states'], axis=0)
            for idx, label_ in enumerate(self.feature_labels):
                df[label_] += states[..., idx].tolist()
        return pd.DataFrame(df)


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def append(self, agent_id, states, rewards, actions, meta={}):

        if self.replay_buffer_length > self.training_params['max_replay_buffer_length']:

            # reset the replay buffer
            self.replay_buffer_length = 0
            self.replay_buffer = {}

        if not agent_id in self.replay_buffer:
            self.replay_buffer[agent_id] = dict(states=[], rewards=[], actions=[], meta=[])
        
        self.replay_buffer[agent_id]['states'].append(states)
        self.replay_buffer[agent_id]['rewards'].append(rewards)
        self.replay_buffer[agent_id]['actions'].append(actions)
        self.replay_buffer[agent_id]['meta'].append(meta)

        self.replay_buffer_length += 1


    def handler(self, data, obs):
        try: 
            states_dict = get_features(obs)
        except Exception as e:
            self.error_obs.append((e, obs))

            raise e

        if self.feature_labels is None:  
            self.feature_labels = list(states_dict.keys())

        states = np.array(list(states_dict.values()))
        random = (self.epsilon > np.random.rand(1)[0])

        q_values = self.model(states[None, ...]).numpy()[0]
        if random: 
            action = int(np.random.randint(4))
        else: 
            
            action = int(np.argmax(q_values))
            
        if self.to_record: 
            
            meta = dict(training_step=self.training_step, random=random, epsilon=self.epsilon, obs=obs)
            for i in range(4):
                meta[f'q{i}'] = q_values[i]

            self.append(data['receiver_id'], states=states, rewards=data['message']['rewards'], actions=action, meta=meta)

        self.total_step += 1 
        if self.training_params['epsilon_random_step'] < self.total_step: 

            # reduce the chance to perform a random action after a certain amount of steps
            self.epsilon -= self.training_params['epsilon_interval'] / self.training_params['epsilon_greedy_step']
            self.epsilon = max(self.epsilon, self.training_params['epsilon_min'])

        
        return action
