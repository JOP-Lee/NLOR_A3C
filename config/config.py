import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.GAMMA = 0.99

cfg.MOVE_FACTOR = 0.2

cfg.SCALE_FACTOR = 0.1

cfg.HISTORY_LENGTH = 50

cfg.NUM_ACTIONS = 9

cfg.PREVENT_STUCK = True
