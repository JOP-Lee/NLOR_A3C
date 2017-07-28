#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import random, time
import threading, multiprocessing
import numpy as np
from tqdm import tqdm
from six.moves import queue

from tensorpack import *
from tensorpack.predict import get_predict_func
from tensorpack.utils.concurrency import *
from tensorpack.utils.stat import  *
from tensorpack.callbacks import *

global get_player
get_player = None

def play_one_episode(player, func, verbose=False):
    def f(s):
        spc = player.get_action_space()
        prob, state = func([[s]])
        player.rnn_state = state[0]
        act = prob[0].argmax()
        if random.random() < 0.001:
            act = spc.sample()
        if verbose:
            print(act)
        return act
    return np.mean(player.play_one_episode(f))

def play_model(cfg):
    player = get_player(viz=0.01)
    predfunc = get_predict_func(cfg)
    while True:
        score = play_one_episode(player, predfunc)
        print("Total:", score)

def eval_with_funcs(predict_funcs, nr_eval):
    class Worker(StoppableThread):
        def __init__(self, func, queue):
            super(Worker, self).__init__()
            self._func = func
            self.q = queue

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def run(self):
            p = get_player(train=False)
	    def get_predict(s):
		return  self.func([[s]])

	    cnt = 0
	    det_cnt = 0
	    iou_cnt = 0
	    for _ in tqdm(range(p.dataset.IMG_NUM)):
	      detected = False
	      ioued = False
	      o = False
	      while not o:
		      prob, state = get_predict(p.current_state())
		      prob = prob[0]
		      state = state[0]
		      action = np.random.choice(len(prob), p=prob)
		      if p.current_iou() >= 0.5:
			if not ioued:
			  iou_cnt += 1
			  ioued = True
		      if action == 8 and p.current_iou() >= 0.5:
			if not detected:
			  det_cnt += 1
			  detected = True
		      r, o = p.action(action)

		      if o:
			cnt += 1
			detected = False
			ioued = False

	    self.queue_put_stoppable(self.q, [float(det_cnt) / cnt, float(iou_cnt) / cnt])

    q = queue.Queue()
    threads = [Worker(f, q) for f in predict_funcs]

    for k in threads:
        k.start()
        time.sleep(0.1) # avoid simulator bugs
    try:
        r, ir = q.get()
        logger.info("Waiting for all the workers to finish the last run...")
        for k in threads: k.join()
    except:
        logger.exception("Eval")
    finally:
        return (r, ir)


def eval_model_multithread(cfg, nr_eval):
    func = get_predict_func(cfg)
    NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
    mean, max = eval_with_funcs([func] * NR_PROC, nr_eval)
    logger.info("Average Score: {}; Max Score: {}".format(mean, max))

class Evaluator(Callback):
    def __init__(self, nr_eval, input_names, output_names):
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names

    def _setup_graph(self):
        self.pred_funcs = [self.trainer.get_predict_func(
            self.input_names, self.output_names)] *1

    def _trigger_epoch(self):
        recall, iou_recall = eval_with_funcs(self.pred_funcs, nr_eval=self.eval_episode)

        self.trainer.write_scalar_summary('recall', recall)
        self.trainer.write_scalar_summary('iou_recall', iou_recall)
