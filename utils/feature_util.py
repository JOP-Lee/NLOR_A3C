#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: feature_util.py
# Author: Fan Wu <jxwufan@gmail.com>

from scipy import sparse
import numpy as np

FILTER_NUM = 2048

def save_sparse_feature(file_path, feature):
  shape = feature.shape

  feature = feature.reshape([shape[0], -1])
  sparse_feature = sparse.csr_matrix(feature)

  np.savez(file_path, data = sparse_feature.data,
            indices = sparse_feature.indices,
            indptr = sparse_feature.indptr,
            shape = sparse_feature.shape )

def load_sparse_feature(file_path, filter_num=FILTER_NUM):

  loader = np.load(file_path)
  sparse_feature = sparse.csr_matrix((loader['data'], loader['indices'],
    loader['indptr']), shape  = loader['shape'])

  feature = sparse_feature.toarray()
  shape = feature.shape
  return feature.reshape([shape[0], -1, filter_num])

if __name__ == "__main__":
  feature =load_sparse_feature("/data/uts411/fan/VOC/VOCdevkit/VOC2007/JPEGImages/000048_vgg.npz",
      512)
  print feature.shape


