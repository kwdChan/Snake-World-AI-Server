import asyncio
import json
import websockets
import numpy as np 
import logging
from policies.dev import policy0
from policies.utils import Inspector

def gd_vector2i_array_parser(vec_txt_list):
    r = []
    for vec_txt_ in vec_txt_list:
        x, y = vec_txt_[1:-1].split(', ')
    
        r.append([int(x), int(y)])
    return np.array(r)




async def main(inspector):
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
            logging.info("incoming")
            message = json.loads(data['message'])

            body = gd_vector2i_array_parser(message['body'])
            edible = gd_vector2i_array_parser(message['edible'])
            inedible = gd_vector2i_array_parser(message['inedible'])
            no_go = gd_vector2i_array_parser(message['no_go'])

            if data['tag'] == "PolicyRemoteControl":
                logging.info("PolicyRemoteControl")
        
                action = policy0(body, edible, inedible, no_go, inspector)

            else:
                # 
                logging.warning("unknown policy")

            res = {}
            res['receiver_id'] = data['receiver_id']
            res['tag'] = data['tag']
            res['message'] = { 'time':message['time']}
            res['message']['action'] = action
            await ws.send(json.dumps(res))

            
    async with websockets.serve(handler, "127.0.0.1", 8001):
        await asyncio.Future()  # run forever
    

if __name__ == "__main__":
    asyncio.run(main(Inspector()))