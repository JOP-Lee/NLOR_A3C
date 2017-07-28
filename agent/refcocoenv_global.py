import time
import cv2
from datasets.refcoco import refcoco
from tensorpack.tfutils.common import get_default_sess_config
from tensorpack.utils import logger

from tensorpack.utils.fs import *
from tensorpack.utils.stat import *
from tensorpack.RL.envbase import RLEnvironment, DiscreteActionSpace
from collections import deque
from config.config import cfg
from utils.image import image_utils
from utils.bbox import *

import numpy as np
import tensorflow as tf
import utils.roi_pooling_layer.roi_pooling_op as roi_pool

from tensorpack.utils.serialize import *

#ACTIONS
ACT_RT = 0 #Right
ACT_LT = 1 #Left
ACT_UP = 2 #Up
ACT_DN = 3 #Down
ACT_TA = 4 #Taller
ACT_FA = 5 #Fatter
ACT_SR = 6 #Shorter
ACT_TH = 7 #Thiner
ACT_TR = 8 #Trigger

RNN_SIZE = 1024

GAMMA = cfg.GAMMA
ALPHA = cfg.MOVE_FACTOR
BETA = cfg.SCALE_FACTOR

NUM_ACTIONS = cfg.NUM_ACTIONS
HISTORY_LENGTH = cfg.HISTORY_LENGTH

PREVENT_STUCK = cfg.PREVENT_STUCK

def history2vec(q):
  """
  Convert action history deque to vector for constructing state
  :param q: deque contains histories
  """
  assert type(q) == deque
  assert q.maxlen == HISTORY_LENGTH

  history = np.array(q)
  res = np.zeros((HISTORY_LENGTH, NUM_ACTIONS))
  if len(q) != 0:
    res[np.arange(len(q)) + (HISTORY_LENGTH - len(q)), history] = 1
  res = res.reshape((HISTORY_LENGTH * NUM_ACTIONS)).astype(np.float32)

  return res

class RefcocoEnv(RLEnvironment):
  def __init__(self, data_name, split, auto_restart=True):
    super(RefcocoEnv, self).__init__()
    self.dataset = refcoco(data_name, split)
    self.data_name = data_name
    self.split = split

    self.reset()

    self.iou = self.current_iou()
    self.last_iou = self.current_iou()
    self.reset_stat()
    self.rwd_counter = StatCounter()
    self.auto_restart = auto_restart

    self.build_roi_pooling_op()

  def build_roi_pooling_op(self):
    config = tf.ConfigProto(device_count = {'GPU': 0})
    self.sess = tf.Session(config=config)
    self.input_feature = tf.placeholder(dtype=tf.float32, shape=[1, None,
                                                                 None, None], name="input_feature")
    self.input_bbox = tf.placeholder(dtype=tf.int32, shape=[1, 5], name="input_bbox")

    self.roi_pooling_op = roi_pool.roi_pool(self.input_feature,
                                            tf.cast(self.input_bbox, tf.float32), 7, 7, 1.0 / 16, "ROI_pool")

  def reset(self):
    self.perm = np.random.permutation(self.dataset.DATA_NUM)
    #self.perm = list(range(self.dataset.DATA_NUM))
    self.sentence_index = 0
    self.total_steps = 0
    self.current_image = None

    self.clear_state_counter()
    self.load_image()

    self.init_state()

  def load_image(self):
    self.data = self.dataset.get_image_data_by_index(self.perm[self.sentence_index])
    data = self.data
    self.image_id = data['file_name']
    self.image_feature = self.dataset.get_feature_map_by_id(self.image_id)
    self.image_feature = np.expand_dims(self.image_feature, 0)
    self.image_vector = self.dataset.get_feature_vector_by_id(self.image_id)
    self.current_image = self.dataset.get_image_by_id(self.image_id)
    self.vector = data['vector']
    self.bbox = data['bbox'].reshape([1, 4]).astype(np.uint16)

    self.image_height = self.current_image.shape[0]
    self.image_width = self.current_image.shape[1]

  def init_state(self):
    #Initialize localization window
    self.x1, self.y1 = 0, 0
    self.x2, self.y2 = self.image_width - 1, self.image_height - 1
    self.iou = 0
    self.last_iou = 0
    self.update_iou()

  def clear_state_counter(self):
    self.total_steps = 0
    self.action_history = deque(maxlen=HISTORY_LENGTH)
    self.rnn_state = np.zeros(RNN_SIZE * 2)

  def next_image(self):
    self.sentence_index += 1
    if (self.sentence_index == self.dataset.DATA_NUM):
      self.reset()
    else :
      self.clear_state_counter()
      self.load_image()
      self.init_state()

  def update_iou(self):
    self.last_iou = self.current_iou()
    if self.current_iou() > self.iou:
        self.iou = self.current_iou()

  def current_iou(self):
    return np.amax(bbox_overlaps(np.array([[self.x1, self.y1, self.x2, self.y2]]).astype(np.float64),
                                 self.bbox.astype(np.float64)))

  def draw_state(self):
    image = self.current_image.copy()
    for box in self.bbox:
      image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))
    image = cv2.rectangle(image, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0))
    cv2.imshow("current", image)
    #cv2.waitKey(1)
    print "Current window:", self.x1, self.y1, self.x2, self.y2
    print "Current Boxes:", self.bbox
    print "Current IOU:", self.current_iou()
    print "High IOU:", self.iou
    print "Last IOU:", self.last_iou
    print "Total steps", self.total_steps
    print "Current index", self.sentence_index
    print self.data['sentence']
    print self.stats['score']

  def window_height(self):
    return self.y2 - self.y1 + 1

  def window_width(self):
    return self.x2 - self.x1 + 1

  def current_state(self):

    window_feature = self.sess.run(self.roi_pooling_op, feed_dict={
      self.input_feature: self.image_feature,
      self.input_bbox: np.array([[0, self.x1, self.y1, self.x2, self.y2]]),
    })[0]

    window_feature = np.squeeze(window_feature)
    window_feature = np.mean(window_feature, (0, 1))

    window_vec = [float(self.x1)/(self.image_width-1),
                 float(self.y1)/(self.image_height-1),
                 float(self.x2)/(self.image_width-1),
                 float(self.y2)/(self.image_height-1),
        float((self.x2-self.x1+1)*(self.y2-self.y1+1))/self.image_height/self.image_width]

    return np.concatenate((window_feature.reshape([-1]), self.image_vector, self.vector,
      history2vec(self.action_history), window_vec, self.rnn_state))

  def get_action_space(self):
    return DiscreteActionSpace(NUM_ACTIONS)

  def action(self, act):

    assert act >= ACT_RT and act <= ACT_TR
    self.action_history.append(act)

    if act <= ACT_DN:
      delta_w = int(ALPHA * self.window_width())
      delta_h = int(ALPHA * self.window_height())
    else:
      delta_w = int(BETA * self.window_width())
      delta_h = int(BETA * self.window_height())

    if PREVENT_STUCK:
      if (delta_h == 0):
        delta_h = 1
      if (delta_w == 0):
        delta_w = 1

    #Do the corresponding action to the window
    if act == ACT_RT:
      self.x1 += delta_w
      self.x2 += delta_w
    elif act == ACT_LT:
      self.x1 -= delta_w
      self.x2 -= delta_w
    elif act == ACT_UP:
      self.y1 -= delta_h
      self.y2 -= delta_h
    elif act == ACT_DN:
      self.y1 += delta_h
      self.y2 += delta_h
    elif act == ACT_TA:
      self.y1 -= delta_h
      self.y2 += delta_h
    elif act == ACT_FA:
      self.x1 -= delta_w
      self.x2 += delta_w
    elif act == ACT_SR:
      self.y1 += delta_h
      self.y2 -= delta_h
    elif act == ACT_TH:
      self.x1 += delta_w
      self.x2 -= delta_w
    elif act == ACT_TR:
      pass
    else:
      raise NotImplemented

    # ensure bbox inside image
    if self.x1 < 0:
      self.x1 = 0
    if self.y1 < 0:
      self.y1 = 0
    if self.x2 >= self.image_width:
      self.x2 = self.image_width - 1
    if self.y2 >= self.image_height:
      self.y2 = self.image_height - 1

    # ensure p1 <= p2
    if PREVENT_STUCK:
      x1 = min(self.x1, self.x2)
      x2 = max(self.x1, self.x2)
      y1 = min(self.y1, self.y2)
      y2 = max(self.y1, self.y2)
      self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    #TODO: remove assert
    assert self.x1 <= self.x2
    assert self.y1 <= self.y2

    #Get reward
    if act < ACT_TR:
      if (self.iou < self.current_iou()):
        reward = self.current_iou()
      else:
        reward = -0.05
    else:
      if self.current_iou() < 0.5:
        reward = -1.0
      else:
        reward = 1.0

    reward += (-self.last_iou + GAMMA * self.current_iou())

    self.rwd_counter.feed(reward)

    #Update steps
    self.total_steps += 1

    isOver = False

    if act == ACT_TR:
      isOver = True

    if (self.total_steps >= 200000):
      isOver = True

    if isOver:
      self.finish_episode()
      if self.auto_restart:
        self.restart_episode()

    #Update iou
    self.update_iou()

    return reward, isOver

  def restart_episode(self):
    self.next_image()

  def finish_episode(self):
    self.stats['score'].append(self.rwd_counter.sum)
    self.rwd_counter.reset()

class RefcocoEnvEval(RefcocoEnv):
  def __init__(self, data_name, split, idx_start, idx_end):
    self.idx_start = idx_start
    self.idx_end = idx_end
    super(RefcocoEnvEval, self).__init__(data_name, split)

  def reset(self):
    self.perm = range(self.dataset.DATA_NUM)
    self.sentence_index = self.idx_start
    self.total_steps = 0
    self.current_image = None

    self.clear_state_counter()
    self.load_image()

    self.init_state()

  def next_image(self):
    self.sentence_index += 1
    if (self.sentence_index == self.idx_end):
      pass
    else :
      self.clear_state_counter()
      self.load_image()
      self.init_state()

if __name__ == '__main__':
  env = RefcocoEnv("refcoco", "train")
  num = env.get_action_space().num_actions()

  from tensorpack.utils import *
  rng = get_rng(num)
  cnt = 0
  #env.draw_state()
 # cv2.waitKey()
  print "Feature:", env.current_state().shape
  input()

  while True:
    act = rng.choice(range(num))
    print act
    r, o = env.action(act)
    print r, o
    if o:
      print env.sentence_index
      print env.stats['score']
    print "Feature:", env.current_state()[-5:]
    print ""
    cv2.waitKey(100)

#    print ""
