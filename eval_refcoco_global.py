#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: eval_refcoco_global.py
# Author: Fan Wu <jxwufan@gmail.com>

import numpy as np
import tensorflow as tf
import os, sys, re, time
import random
import argparse
import six
import zmq
from tqdm import tqdm

from tensorpack import *
from tensorpack.RL import *
from tensorpack.predict.common import PredictConfig
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.RL.simulator import SimulatorProcess, SimulatorProcessBase

import numpy as np
import tensorflow as tf
import os, sys, re, time
import random
import uuid
import argparse
import multiprocessing, threading
from collections import deque
import six
from six.moves import queue
from threading import Thread

from tensorpack import *
from tensorpack.utils.concurrency import *
from tensorpack.utils.serialize import *
from tensorpack.utils.timer import *
from tensorpack.utils.stat import  *
from tensorpack.tfutils import symbolic_functions as symbf

from tensorpack.RL import *
import utils.common as common
from utils.common import (play_model, Evaluator, eval_model_multithread)

from agent.refcocoenv_global import RefcocoEnvEval
from config.config import cfg

import tensorflow.contrib.slim as slim

from utils.common import play_one_episode

import cv2
from tensorflow.contrib.rnn.python.ops.rnn_cell import LayerNormBasicLSTMCell

from datasets.refcoco import refcoco

PIPE_DIR = os.environ.get('TENSORPACK_PIPEDIR', '.').rstrip('/')
namec2s = 'ipc://{}/sim-c2s-{}'.format(PIPE_DIR, "eval")
names2c = 'ipc://{}/sim-s2c-{}'.format(PIPE_DIR, "eval")

NUM_ACTIONS = 9
HISTORY_LENGTH = 50
ENV_NAME = None
SPLIT_NAME = None

RNN_SIZE = 1024

LANG_VECTOR_SIZE = 4800
WIN_VECTOR_SIZE = 5
VISUAL_SIZE = 4096

NUM_SIM = 20

def get_player(start_idx, end_idx):
  return RefcocoEnvEval(ENV_NAME, SPLIT_NAME, start_idx, end_idx)

class MySimulatorWorker(SimulatorProcessBase):

  def __init__(self, idx, pipe_c2s, pipe_s2c, start_idx, end_idx):
    super(MySimulatorWorker, self).__init__(idx)
    self.pipe_c2s = pipe_c2s
    self.pipe_s2c = pipe_s2c
    self.start_idx = start_idx
    self.end_idx = end_idx
    self.num_sent = end_idx - start_idx
    self.cnt = 0
    self.cnt_det = 0
    self.cnt_reach = 0
    self.idx = idx

  def connect(self):
    #Set pipe to master
    context = zmq.Context()
    self.pipe_c2s_socket = context.socket(zmq.PUSH)
    self.pipe_c2s_socket.setsockopt(zmq.IDENTITY, self.identity)
    #self.pipe_c2s_socket.set_hwm(60)
    self.pipe_c2s_socket.connect(self.pipe_c2s)

    self.pipe_s2c_socket = context.socket(zmq.DEALER)
    self.pipe_s2c_socket.setsockopt(zmq.IDENTITY, self.identity)
    #self.pipe_s2c_socket.set_hwm(5)
    self.pipe_s2c_socket.connect(self.pipe_s2c)

  def run(self):
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    self.connect()
    #Build player after connected
    player = self._build_player()

    reward, i1isOver = 0, False
    for self.cnt in tqdm(range(self.num_sent), desc="Simulator-"+str(self.idx), position=int(self.idx)):
      detected = False
      reached = False
      while True:
        state = player.current_state()
        self.pipe_c2s_socket.send(dumps((self.identity, "Run", state)), copy=False)
        action, rnn_state = loads(self.pipe_s2c_socket.recv(copy=False).bytes)
        player.rnn_state = rnn_state

        if player.current_iou() > 0.5 and not reached:
          self.cnt_reach += 1
          reached = True

        if action == 8 and player.current_iou() > 0.5 and not detected:
          self.cnt_det += 1
          detected = True
        _, isOver = player.action(action)
        if isOver:
          break

    mt = self.pipe_c2s_socket.send(dumps((self.identity, "Over", self.cnt_det,
      self.cnt_reach, self.num_sent)), copy=False, track=True)

    mt.wait()


  def _build_player(self):
    return get_player(self.start_idx, self.end_idx)

def evaluate(cfg):

  context = zmq.Context()

  c2s_socket = context.socket(zmq.PULL)
  c2s_socket.bind(namec2s)
  c2s_socket.set_hwm(60)
  s2c_socket = context.socket(zmq.ROUTER)
  s2c_socket.bind(names2c)
  s2c_socket.set_hwm(60)

  cnt = 0
  cnt_det = 0
  cnt_reach = 0

  func = get_predict_func(cfg)
  def get_predict(s):
    return  func([s])

  buf_size = NUM_SIM

  while buf_size != 0:
    state_buf = []
    identity_buf = []
    while len(state_buf) < buf_size:

      rec = loads(c2s_socket.recv(copy=False).bytes)
      identity = rec[0]
      status = rec[1]
      if status == "Over":
        buf_size -= 1
        cnt_det += rec[2]
        cnt_reach += rec[3]
        cnt += rec[4]
      elif status == "Run":
        identity_buf.append(identity)
        state_buf.append(rec[2])
      else:
        raise NotImplementedError

    if buf_size == 0:
      break
    probs, states = get_predict(state_buf)

    for i in range(buf_size):
      identity = identity_buf[i]
      prob = probs[i]
      state = states[i]
      action = np.random.choice(len(prob), p=prob)
      s2c_socket.send_multipart([identity, dumps([action, state])], copy=False)

  for i in range(100):
    print ""
  print "total det:", cnt_det
  print "total reach", cnt_reach
  print "total sent", cnt
  print "det percent:", float(cnt_det) / cnt
  print "reach percent:", float(cnt_reach) / cnt

class Model(ModelDesc):
  def _get_input_vars(self):
    assert NUM_ACTIONS is not None
    return [InputVar(tf.float32, (None, WIN_VECTOR_SIZE + VISUAL_SIZE + LANG_VECTOR_SIZE + HISTORY_LENGTH*NUM_ACTIONS + RNN_SIZE*2), 'state'),
        InputVar(tf.int64, (None,), 'action'),
        InputVar(tf.float32, (None,), 'futurereward') ]

  def _get_NN_prediction(self, state):
    visual = state[:,:VISUAL_SIZE]
    lang = state[:,VISUAL_SIZE:VISUAL_SIZE+LANG_VECTOR_SIZE]
    lang = slim.fully_connected(lang, VISUAL_SIZE, scope='fc/lang')
    other = state[:,VISUAL_SIZE+LANG_VECTOR_SIZE: WIN_VECTOR_SIZE+VISUAL_SIZE+LANG_VECTOR_SIZE+HISTORY_LENGTH*NUM_ACTIONS]
    rnn_state = state[:, WIN_VECTOR_SIZE+VISUAL_SIZE+LANG_VECTOR_SIZE+HISTORY_LENGTH*NUM_ACTIONS:]

    c = rnn_state[:, :RNN_SIZE]
    h = rnn_state[:, RNN_SIZE:]
    rnn_state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

    context = tf.mul(visual, lang)
    context = tf.nn.l2_normalize(context, 1)

    l = tf.concat(1, [context, other])
    l = slim.fully_connected(l, 1024, scope='fc/fc1')
    l = slim.fully_connected(l, 1024, scope='fc/fc2')

    rnn_cell = LayerNormBasicLSTMCell(RNN_SIZE)
    l, rnn_state = rnn_cell(l, rnn_state)

    c, h = rnn_state
    rnn_state = tf.concat(1, [c, h], name='rnn_state')

    policy = slim.fully_connected(l, 9, activation_fn=None, scope='fc/fc-pi')
    value = slim.fully_connected(l, 1, activation_fn=None, scope='fc/fc-v')

    return policy, value

  def _build_graph(self, inputs):
    state, action, futurereward = inputs
    policy, self.value = self._get_NN_prediction(state)
    self.value = tf.squeeze(self.value, [1], name='pred_value') # (B,)
    self.logits = tf.nn.softmax(policy, name='logits')

    expf = tf.get_variable('explore_factor', shape=[],
        initializer=tf.constant_initializer(1), trainable=False)
    logitsT = tf.nn.softmax(policy * expf, name='logitsT')
    is_training = get_current_tower_context().is_training
    if not is_training:
      return
    log_probs = tf.log(self.logits + 1e-6)

    log_pi_a_given_s = tf.reduce_sum(
        log_probs * tf.one_hot(action, NUM_ACTIONS), 1)
    advantage = tf.sub(tf.stop_gradient(self.value), futurereward, name='advantage')
    policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage, name='policy_loss')
    xentropy_loss = tf.reduce_sum(
        self.logits * log_probs, name='xentropy_loss')
    value_loss = tf.nn.l2_loss(self.value - futurereward, name='value_loss')

    pred_reward = tf.reduce_mean(self.value, name='predict_reward')
    advantage = symbf.rms(advantage, name='rms_advantage')
    summary.add_moving_summary(policy_loss, xentropy_loss, value_loss, pred_reward, advantage)
    entropy_beta = tf.get_variable('entropy_beta', shape=[],
        initializer=tf.constant_initializer(0.01), trainable=False)
    self.cost = tf.add_n([policy_loss, xentropy_loss * entropy_beta, value_loss])
    self.cost = tf.truediv(self.cost,
        tf.cast(tf.shape(futurereward)[0], tf.float32),
        name='cost')

procs = []
#TODO start multiprocess env before env to prevent use gpu
def start_simulators():
#Get num of sentences
  dataset = refcoco(ENV_NAME, SPLIT_NAME)
  data_num = dataset.DATA_NUM
  if data_num % NUM_SIM != 0:
    sent_per_sim = data_num // NUM_SIM + 1
  else:
    sent_per_sim = data_num // NUM_SIM

  start_idx = 0
  for i in range(NUM_SIM):
    end_idx = start_idx + sent_per_sim
    print "For i=", i, start_idx, end_idx
    if end_idx > data_num:
      end_idx = data_num
    procs.append(MySimulatorWorker(i, namec2s, names2c, start_idx, end_idx))
    start_idx = end_idx

  for i in range(len(procs)):
    p = procs[i]
    p.start()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', required = True)
  parser.add_argument('--load', help='load model', required=True)
  parser.add_argument('--env', help='environment name', required=True)
  parser.add_argument('--split', help='split name', required=True)
  args = parser.parse_args()

  ENV_NAME = args.env
  SPLIT_NAME = args.split
  assert ENV_NAME
  assert SPLIT_NAME
  logger.info("Environment Name: {}".format(ENV_NAME))
  logger.info("Split Name: {}".format(SPLIT_NAME))

  if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

  start_simulators()

  cfg = PredictConfig(
      model=Model(),
      session_init=SaverRestore(args.load),
      input_names=['state'],
      output_names=['logits', 'rnn_state'])

  eval_thread = Thread(target=evaluate, args=(cfg,))
  eval_thread.start()

  for p in procs:
    p.join()
  eval_thread.join()
