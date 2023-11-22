from my_types import Action
from .utils import to_sparse
import matplotlib.pyplot as plt


VISION_XBOUNDS = (-20, 20)
VISION_YBOUNDS = (-20, 20)

def policy0(body, edible, inedible, no_go, inspector):

    inspector.data['no_go'] = to_sparse(no_go, VISION_XBOUNDS, VISION_YBOUNDS)
    inspector.data['edible'] = to_sparse(edible, VISION_XBOUNDS, VISION_YBOUNDS)
    inspector.data['inedible'] = to_sparse(inedible, VISION_XBOUNDS, VISION_YBOUNDS)
    inspector.data['body'] = to_sparse(body, VISION_XBOUNDS, VISION_YBOUNDS)





    

    return Action.STAY