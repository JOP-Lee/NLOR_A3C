#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train_refcoco_nornn.py
# Author: Fan Wu <jxwufan@gmail.com>
from abc import ABCMeta

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

from tensorpack import *
from tensorpack.RL.simulator import SimulatorProcess, SimulatorProcessBase
from tensorpack.utils.concurrency import *
from tensorpack.utils.serialize import *
from tensorpack.utils.timer import *
from tensorpack.utils.stat import  *
from tensorpack.tfutils import symbolic_functions as symbf

from tensorpack.RL import *
import utils.common as common
from utils.common import (play_model, Evaluator, eval_model_multithread)

from agent.refcocoenv_nornn import RefcocoEnv
from config.config import cfg

import tensorflow.contrib.slim as slim
from tensorpack.callbacks.base import Callback
from tensorpack.tfutils.common import get_global_step
from tensorpack.utils.serialize import *
import multiprocessing as mp
import zmq

from utils.image import image_utils
import os

GAMMA = cfg.GAMMA

LOCAL_TIME_MAX = 5
STEP_PER_EPOCH = 20000
EVAL_EPISODE = 50
BATCH_SIZE = 128
SIMULATOR_PROC = 50
PREDICTOR_THREAD_PER_GPU = 2
PREDICTOR_THREAD = None
EVALUATE_PROC = min(multiprocessing.cpu_count() // 2, 20)

NUM_ACTIONS = cfg.NUM_ACTIONS
HISTORY_LENGTH = cfg.HISTORY_LENGTH
ENV_NAME = None

global_step = None

APPRENTICESHIP_LR = False

name_base = str(uuid.uuid1())[:6]
PIPE_DIR = os.environ.get('TENSORPACK_PIPEDIR', '.').rstrip('/')
namec2s = 'ipc://{}/sim-c2s-{}'.format(PIPE_DIR, name_base)
names2c = 'ipc://{}/sim-s2c-{}'.format(PIPE_DIR, name_base)

VISUAL_LEN = 2048
SPATIAL_LEN = 5
HISTORY_LEN = 450
LANG_LEN = 4800

def softmax(logit):
  exp = np.exp(logit)
  return exp / np.sum(exp)


def get_player(viz=False, train=False, dumpdir=None):
  pl = RefcocoEnv(ENV_NAME, "train")

  global NUM_ACTIONS
  NUM_ACTIONS = pl.get_action_space().num_actions()

  return pl

class MySimulatorWorker(SimulatorProcessBase):

  def __init__(self, idx, pipe_c2s, pipe_s2c):
    super(MySimulatorWorker, self).__init__(idx)
    self.idx = idx
    self.pipe_c2s = pipe_c2s
    self.pipe_s2c = pipe_s2c

  def connect(self):
    #Set pipe to master
    context = zmq.Context()
    self.pipe_c2s_socket = context.socket(zmq.PUSH)
    self.pipe_c2s_socket.setsockopt(zmq.IDENTITY, self.identity)
    self.pipe_c2s_socket.set_hwm(2)
    self.pipe_c2s_socket.connect(self.pipe_c2s)

    self.pipe_s2c_socket = context.socket(zmq.DEALER)
    self.pipe_s2c_socket.setsockopt(zmq.IDENTITY, self.identity)
    #self.pipe_s2c_socket.set_hwm(5)
    self.pipe_s2c_socket.connect(self.pipe_s2c)

  def run(self):
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    np.random.seed(int(self.idx))
    self.connect()
    #Build player after connected
    player = self._build_player()

    state = player.current_state()
    reward, isOver = 0, False
    while True:
      self.pipe_c2s_socket.send(dumps(
        (self.identity, state, reward, isOver)),
        copy=False)
      action = loads(self.pipe_s2c_socket.recv(copy=False).bytes)[0]
      reward, isOver = player.action(action)
      state = player.current_state()

  def _build_player(self):
    return get_player(train=True)

class Model(ModelDesc):
  def _get_input_vars(self):
    assert NUM_ACTIONS is not None
    return [InputVar(tf.float32, (None, SPATIAL_LEN + VISUAL_LEN  + LANG_LEN + HISTORY_LENGTH*NUM_ACTIONS), 'state'),
        InputVar(tf.int64, (None,), 'action'),
        InputVar(tf.float32, (None,), 'futurereward') ]

  def _get_NN_prediction(self, state):
    visual = state[:,:VISUAL_LEN]
    lang = state[:,VISUAL_LEN:VISUAL_LEN+LANG_LEN]
    lang = slim.fully_connected(lang, VISUAL_LEN, scope='fc/lang')
    other = state[:,VISUAL_LEN+LANG_LEN: SPATIAL_LEN+VISUAL_LEN+LANG_LEN+HISTORY_LENGTH*NUM_ACTIONS]

    context = tf.mul(visual, lang)
    context = tf.nn.l2_normalize(context, 1)

    l = tf.concat(1, [context, other])
    l = slim.fully_connected(l, 1024, scope='fc/fc1')
    l = slim.fully_connected(l, 1024, scope='fc/fc2')

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

  def get_gradient_processor(self):
    return [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1)),
        SummaryGradient()]

class MySimulatorMaster(SimulatorMaster, Callback):
  def __init__(self, pipe_c2s, pipe_s2c, model):
    super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c)
    self.M = model
    self.queue = queue.Queue(maxsize=BATCH_SIZE*8*2)

  def _setup_graph(self):
    self.sess = self.trainer.sess
    self.async_predictor = MultiThreadAsyncPredictor(
        self.trainer.get_predict_funcs(['state'], ['logitsT', 'pred_value',],
        PREDICTOR_THREAD), batch_size=15)
    self.async_predictor.run()

  def _on_state(self, state, ident):
    def cb(outputs):
      distrib, value = outputs.result()

      action = np.random.choice(len(distrib), p=distrib)

      client = self.clients[ident]
      client.memory.append(TransitionExperience(state, action, None, value=value))
      self.send_queue.put([ident, dumps([action,])])
    self.async_predictor.put_task([state], cb)

  def _on_episode_over(self, ident):
    self._parse_memory(0, ident, True)

  def _on_datapoint(self, ident):
    client = self.clients[ident]
    if len(client.memory) == LOCAL_TIME_MAX + 1:
      R = client.memory[-1].value
      self._parse_memory(R, ident, False)

  def _parse_memory(self, init_r, ident, isOver):
    client = self.clients[ident]
    mem = client.memory
    if not isOver:
      last = mem[-1]
      mem = mem[:-1]

    mem.reverse()
    R = float(init_r)
    for idx, k in enumerate(mem):
      R = np.clip(k.reward, -5, 5) + GAMMA * R
      self.queue.put([k.state, k.action, R])

    if not isOver:
      client.memory = [last]
    else:
      client.memory = []

class GlobalStepSetter(Callback):

  def trigger_step(self):
    global global_step
    global_step = get_global_step()

def get_config():
  logger.auto_set_dir()
  M = Model()

  master = MySimulatorMaster(namec2s, names2c, M)
  dataflow = BatchData(DataFromQueue(master.queue), BATCH_SIZE)

  lr = symbf.get_scalar_var('learning_rate', 0.0001, summary=True)
  return TrainConfig(
    dataset=dataflow,
    optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
    callbacks=Callbacks([
      StatPrinter(), ModelSaver(),
      HumanHyperParamSetter('learning_rate', 'hyper.txt'),
      HumanHyperParamSetter('entropy_beta', 'hyper.txt'),
      HumanHyperParamSetter('explore_factor', 'hyper.txt'),
      master,
      StartProcOrThread(master),
#      PeriodicCallback(Evaluator(EVAL_EPISODE, ['state'], ['logits']), 1),
      GlobalStepSetter(),
    ]),
    session_config=get_default_sess_config(0.5),
    model=M,
    step_per_epoch=STEP_PER_EPOCH,
    max_epoch=1000,
  )

if __name__ == '__main__':
  global global_step
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
  parser.add_argument('--load', help='load model')
  parser.add_argument('--env', help='env', required=True)
  parser.add_argument('--task', help='task to perform',
      choices=['play', 'eval', 'train'], default='train')
  args = parser.parse_args()

  ENV_NAME = args.env
  assert ENV_NAME

  procs = [MySimulatorWorker(k, namec2s, names2c) for k in range(SIMULATOR_PROC)]

  ensure_proc_terminate(procs)
  start_proc_mask_signal(procs)

  #p = get_player(); del p  # set NUM_ACTIONS

  if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  if args.task != 'train':
    assert args.load is not None

  if args.task != 'train':
    cfg = PredictConfig(
        model=Model(),
        session_init=SaverRestore(args.load),
        input_names=['state'],
        output_names=['logits'])
    if args.task == 'play':
      play_model(cfg)
    elif args.task == 'eval':
      eval_model_multithread(cfg, EVAL_EPISODE)
  else:
    if args.gpu:
      nr_gpu = get_nr_gpu()
      if nr_gpu > 1:
        predict_tower = range(nr_gpu)[-nr_gpu/2:]
      else:
        predict_tower = [0]
      PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
      train_tower = range(nr_gpu)[:-nr_gpu/2] or [0]
      logger.info("[BA3C] Train on gpu {} and infer on gpu {}".format(
        ','.join(map(str, train_tower)), ','.join(map(str, predict_tower))))
    else:
      nr_gpu = 0
      PREDICTOR_THREAD = 1
      predict_tower = [0]
      train_tower = [0]
    config = get_config()
    if args.load:
      config.session_init = SaverRestore(args.load)
      global_step = int(args.load.split('-')[-1])
    else:
      #config.session_init = SaverRestore(cfg.CKPT_PATH)
      global_step = 0
    config.tower = train_tower
    AsyncMultiGPUTrainer(config, predict_tower=predict_tower).train()
