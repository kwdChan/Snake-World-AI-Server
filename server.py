import asyncio
import json
import websockets
import numpy as np 
import logging
from my_types import Action

#from policies.deep_q_learning_pre_extracted import DQLModelPreExtracted

def gd_vector2i_array_parser(vec_txt_list):
    r = []
    for vec_txt_ in vec_txt_list:
        x, y = vec_txt_[1:-1].split(', ')
    
        r.append([int(x), int(y)])
    return np.array(r)

def get_obs(message):
    body = gd_vector2i_array_parser(message['body'])
    edible = gd_vector2i_array_parser(message['edible'])
    inedible = gd_vector2i_array_parser(message['inedible'])
    no_go = gd_vector2i_array_parser(message['no_go'])
    return {'body': body, 'edible': edible, 'inedible': inedible, 'no_go': no_go}




async def main(model_byName):
    async def handler(ws):
        """
        incoming messages must be JSON with the following keys: 
        - receiver_id
        - tag
        - message
            - body
            - edible
            - inedible
            - no_go
            - time
        """
        while True:
            data = json.loads(await ws.recv())
            logging.debug("incoming")
            message = json.loads(data['message'])

            obs = get_obs(message)

            model = model_byName.get(data['tag'])

            if not model: 
                logging.warning('model name not found: %s '%data['tag'])
                action = Action.STAY
            else: 

                agent = model.get_agent(data['receiver_id'])
                action = agent(obs, message['rewards'])
            

            res = {}
            res['receiver_id'] = data['receiver_id']
            res['tag'] = data['tag']
            res['message'] = { 'time':message['time']}
            res['message']['action'] = action
            await ws.send(json.dumps(res))

            
    async with websockets.serve(handler, "127.0.0.1", 8001):
        # await asyncio.sleep(3)
        await asyncio.Future()  # run forever
