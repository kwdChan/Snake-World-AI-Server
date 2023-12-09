import plotly.express as px
import plotly.graph_objects as go
from policies.utils import to_sparse
def show_obs(df, agent_id, obs_type, starting_index=0, xrange=(-20, 20), yrange=(-20, 20)):


    df = df.query(f"agent_id=='{agent_id}'")

    fig = go.FigureWidget([go.Heatmap()], layout=dict(width=450, height=450))

    # so that this value can modified by the increment and decrement functions
    data = {"index": starting_index}

    def increment():
        data['index'] += 1
        update()
    
    def decrement():
        data['index'] -= 1
        update()

    
    def update():
        fig.data[0].z = ((to_sparse(df['meta'].iloc[data['index']]['obs'][obs_type], xrange, yrange))).T

    return fig, increment, decrement




