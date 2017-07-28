#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: run_refcoco_nornn.py
# Author: Fan Wu <jxwufan@gmail.com>

import numpy as np
import tensorflow as tf
import os, sys, re, time
import random
import argparse
import six

from tensorpack import *
from tensorpack.RL import *
from tensorpack.predict.common import PredictConfig
from tensorpack.tfutils.sessinit import SaverRestore

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

from utils.inception_v3 import *
import tensorflow.contrib.slim as slim

from utils.common import play_one_episode

import cv2

LOCAL_TIME_MAX = 5
STEP_PER_EPOCH = 6000
EVAL_EPISODE = 50
BATCH_SIZE = 128
SIMULATOR_PROC = 50
PREDICTOR_THREAD_PER_GPU = 2
PREDICTOR_THREAD = None
EVALUATE_PROC = min(multiprocessing.cpu_count() // 2, 20)

NUM_ACTIONS = 9
HISTORY_LENGTH = 50
ENV_NAME = None
SPLIT_NAME = None

VISUAL_LEN = 2048
SPATIAL_LEN = 5
HISTORY_LEN = 450
LANG_LEN = 4800

def get_player(viz=False, train=False, dumpdir=None):
  pl = RefcocoEnv(ENV_NAME, SPLIT_NAME)

  global NUM_ACTIONS
  NUM_ACTIONS = pl.get_action_space().num_actions()

  return pl
common.get_player = get_player

class Model(ModelDesc):
  def _get_input_vars(self):
    assert NUM_ACTIONS is not None
    return [InputVar(tf.float32, (None, SPATIAL_LEN + VISUAL_LEN + LANG_LEN + HISTORY_LENGTH*NUM_ACTIONS), 'state'),
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

def run_submission(cfg):
  p = get_player()
  func = get_predict_func(cfg)
  def get_predict(s):
    return  func([[s]])

  cnt = 0
  detected = False
  ioued = False
  det_cnt = 0
  iou_cnt = 0
  high_prob = 0
  high_iou = 0
  high_cnt = 0
  #p.draw_state()
  #cv2.waitKey()
  while True:
      prob = get_predict(p.current_state())
      prob = prob[0][0]
      action = np.random.choice(len(prob), p=prob)
      #action = np.argmax(prob)
      #print "action", action
      #if action == 8:
      #  print "Iou", p.current_iou()
      #  p.draw_state()
      #  cv2.waitKey()
     # if  action == 8:
     #   print p.current_iou()
      if p.current_iou() >= 0.5:
        if not ioued:
          iou_cnt += 1
          ioued = True
          #p.draw_state()
          #cv2.waitKey()
      if action == 8 and p.current_iou() >= 0.5:
        if not detected:
          det_cnt += 1
          detected = True
  #        p.draw_state()
  #        cv2.waitKey()
      if high_prob < prob[8]:
        high_prob = prob[8]
        high_iou = p.current_iou()
      r, o = p.action(action)
      #p.draw_state()
      #input()
      #cv2.waitKey()

      if o:
        cnt += 1
        if high_iou > 0.5:
          high_cnt += 1
        detected = False
        ioued = False
        print
        print float(det_cnt) / cnt
        print float(iou_cnt) / cnt
        print float(high_cnt) / cnt
        print high_iou
        print cnt
        high_prob = 0
        high_iou = 0
      if cnt == p.dataset.DATA_NUM:
        break

  print float(det_cnt) / cnt



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
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
  p = get_player(); del p  # set NUM_ACTIONS

  if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

  cfg = PredictConfig(
      model=Model(),
      session_init=SaverRestore(args.load),
      input_names=['state'],
      output_names=['logits'])
  run_submission(cfg)
