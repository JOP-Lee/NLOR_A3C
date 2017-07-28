#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: refcoco.py
# Author: Fan Wu <jxwufan@gmail.com>
import numpy as np
import deepdish as dd
import os
import cv2
from tqdm import tqdm

import utils.feature_util as fu
import cPickle as pickle

IMG_DIR = "preprocessed_images"
FEATURE_DIR = "feature_maps"
VECTOR_DIR = "feature_vectors"
H5_DIR = "h5"

#TODO
#REFCOCO_PATH = "/data/uts411/fan/refer"

class refcoco(object):
  def __init__(self, dataset, split):
    self.H5_PATH = os.path.join(REFCOCO_PATH, H5_DIR, dataset, split + "_data.h5")
    if not os.path.exists(self.H5_PATH):
      print "___________________________________________________________________________________________________________________"
      print self.H5_PATH, "not exist!!!"
      print "___________________________________________________________________________________________________________________"
      raise NotImplementedError

    self.DATA_NUM = dd.io.load(self.H5_PATH, "/NUM")

  def data_num(self):
    return self.DATA_NUM

  def get_image_data_by_index(self, index):
    return  dd.io.load(self.H5_PATH, "/data/i" + str(index))

  def get_image_by_id(self, id):
    id = str(id)
    image_path = os.path.join(REFCOCO_PATH, IMG_DIR, id + ".jpg")
    image = cv2.imread(image_path)

    return image

  def get_feature_vector_by_id(self, id):
    id = str(id)
    vector_path = os.path.join(REFCOCO_PATH, VECTOR_DIR, id + ".npy")
    vector = np.load(vector_path)

    return vector

  def get_feature_map_by_id(self, id):
    id = str(id)
    feature_path = os.path.join(REFCOCO_PATH, FEATURE_DIR, id + ".npz")
    feature = fu.load_sparse_feature(feature_path)

    return feature

def test():
  vrd = refcoco("refcocog", "train")
  data = vrd.get_image_data_by_index(4)

  print data
  print vrd.get_feature_map_by_id(data["file_name"])
  img= vrd.get_image_by_id(data["file_name"])
  bbox = data['bbox']
  im = img.copy()
  cv2.rectangle(im, tuple(bbox[0:2]), tuple(bbox[2:]), (255,255,0))

  cv2.imshow("test", im)
  cv2.waitKey()

if __name__ == "__main__":
  test()

