from typing import Literal

import pandas as pd
import numpy as np

import holoviews as hv

pdIdx = pd.IndexSlice

def unpack_meta(df:pd.DataFrame, keys=[]):
    df = df[['meta']].apply(lambda x: [x['meta'][k] for k in keys], result_type='expand', axis=1)
    df.columns = keys
    return df

class ReplayBufferAnalysis:
    def __init__(self, model, metakey_to_unpack=['training_step', 'random', 'epsilon', 'q0', 'q1', 'q2', 'q3']):
        
        replay_buffer_df = model.replay_buffer_df()


        replay_buffer_df[metakey_to_unpack] = unpack_meta(replay_buffer_df, metakey_to_unpack)


        replay_buffer_df['1000steps'] = replay_buffer_df['training_step'].apply(lambda x: np.floor(x//1000)) # why did i use np.floor?
        replay_buffer_df['100steps'] = replay_buffer_df['training_step'].apply(lambda x: np.floor(x//100))
        replay_buffer_df['shifted_rewards'] = replay_buffer_df['rewards'].shift(-1).fillna(0)
        
        self.model = model
        self.df: pd.DataFrame = replay_buffer_df

    @property
    def agent_list(self):
        return self.df['agent_id'].unique()
    

    def ensure_q_value_calcuated(self):
        if 'shifted_rewards_negative' in self.df:
            return 

        self.df['max_q_discounted'] =  self.df[['q0','q1','q2','q3']].max(1)*self.model.training_params['gamma']
        self.df['correct_q'] = (self.df['max_q_discounted'] + self.df['rewards']).shift(-1)
        self.df['q_of_action'] = self.df[[ 'q0', 'q1','q2','q3', 'actions']].apply(lambda x:x.iloc[int(x.iloc[-1])], axis=1)
        self.df['abs_q_value_difference'] = abs(self.df['q_of_action'] - self.df['correct_q'])  

    
        
        self.df['q_value_difference']= (self.df['q_of_action'] - self.df['correct_q'])

        self.df['shifted_rewards_positive'] = self.df['shifted_rewards'] > 0
        self.df['shifted_rewards_negative'] = self.df['shifted_rewards'] < 0
        self.df['shifted_rewards_sign'] = 'zero'
        self.df.loc[self.df['shifted_rewards_positive'], 'shifted_rewards_sign'] = 'pos'
        self.df.loc[self.df['shifted_rewards_negative'], 'shifted_rewards_sign'] = 'neg'




    def plot_abs_q_value_difference(self, step_agg: Literal['1000steps', '100steps'] = '1000steps', from_step=1):
        self.ensure_q_value_calcuated()
        return hv.Curve(self.df.groupby(step_agg)['abs_q_value_difference'].mean().iloc[from_step:])

    

    def plot_rewards_expectation_error(
            self, 
            reward_sign: Literal['pos', 'neg', 'zeros'], 
            step_agg: Literal['1000steps', '100steps'] = '100steps', 
            from_step=1, 
            title='title',
            
            ):

        self.ensure_q_value_calcuated()
        rewards_expectation_error_mean = self.df.groupby([step_agg, 'shifted_rewards_sign'])['q_value_difference'].mean()
        rewards_expectation_error_std = self.df.groupby([step_agg, 'shifted_rewards_sign'])['q_value_difference'].std()
        rewards_expectation_error = pd.concat([rewards_expectation_error_mean, rewards_expectation_error_std], axis=1)
        rewards_expectation_error.columns = ['mean', 'std']


        rewards_expectation_error = rewards_expectation_error.loc[pdIdx[from_step:, reward_sign],:]

        return (
            hv.Spread(
                rewards_expectation_error.fillna(0), 
                step_agg, 
                vdims=['mean', 'std']
                )*
            hv.Curve(rewards_expectation_error, step_agg,'mean')
            .opts(
                color='red', alpha=0.5, aspect=2, fig_size=300, 
                linewidth=1, ylabel='Expectation Error', title=title
            )
        )
    

