import pandas as pd
import numpy as np 

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
    food_L, food_R, food_F = loc_food_directly_n_steps_away(food, n = 3, nothing_value=5 )
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
    
    enemy_R, enemy_L, enemy_F, enemy_B = loc_n_step_away(obs['inedible'], n=2)
    body_R, body_L, body_F, _ = loc_n_step_away(obs['body'], n=2)

    return np.array([food_L, food_R, food_F, body_R, body_L, body_F])

FEATURE_LABELS = ['food_L', 'food_R', 'food_F', 'body_R', 'body_L', 'body_F']

class Recorder:
    def __init__(self):
        """
        fields: 
            action
            [states]
            reward
            obs: objs
            id
        """
        data = dict(action=[], rewards=[], obs=[], agent_id=[])
        for f_ in FEATURE_LABELS:
            data[f_] = []

        self.data = data

    def append(self, obs, rewards, action, agent_id):
        states = get_features(obs)
        new_row = dict(zip(FEATURE_LABELS, states))

        new_row['rewards'] = rewards
        new_row['action'] = action
        new_row['agent_id'] = agent_id
        new_row['obs'] = obs
        assert set(new_row.keys()) == set(self.data.keys())
        for k in new_row:
            self.data[k].append(new_row[k])

    def handler(self, data, obs):
        self.append(obs, data['message']['rewards'], data['message']['action'], data['receiver_id'])
        return 
