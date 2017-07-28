import tensorflow as tf
import numpy as np
import cv2

from tensorpack.tfutils.common import get_default_sess_config

import tensorflow.contrib.slim as slim
from slim.nets.resnet_v1 import *
from slim.nets.vgg import *
from slim.nets.inception_v3 import *

from multiprocessing import Process

RESNET_152_CKPT_PATH  = "/home/fan/resnet/resnet_v1_152.ckpt"
VGG_16_CKPT_PATH = "/home/fan/vgg_16.ckpt"
INCEPTION_v3_CKPT_PATH = "/home/fan/inception_v3.ckpt"

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

NET = "inception"

class image_utils(object):

  def __init__(self):
    self.image_placeholder = tf.placeholder(tf.float32,
                                            shape=(None, None, None, 3))
    self.extract_feature_op = None
    self.sess =  None
  def build_extract_graph(self):
    if NET == "vgg":
      arg_scope = vgg_arg_scope()
      with slim.arg_scope(arg_scope):
        _, end_points = vgg_16(self.image_placeholder)
        self.extract_feature_op = end_points["vgg_16/conv5/conv5_3"]
      saver = tf.train.Saver()
      print "Restore from", VGG_16_CKPT_PATH
      saver.restore(self.sess, VGG_16_CKPT_PATH)

    elif NET == "resnet152":
      arg_scope = resnet_arg_scope()
      with slim.arg_scope(arg_scope):
        self.extract_feature_op = resnet_v1_152(self.image_placeholder,
            global_pool=False,
            output_stride=16)
      saver = tf.train.Saver()
      print "Restore from", RESNET_152_CKPT_PATH
      saver.restore(self.sess, RESNET_152_CKPT_PATH)

    elif NET == "inception":
      arg_scope = inception_v3_arg_scope()
      with slim.arg_scope(arg_scope):
        _, end_points = inception_v3(self.image_placeholder,
            num_classes=1001, is_training=False)
        self.extract_feature_op = end_points["PreLogits"]
      saver = tf.train.Saver()
      print "Restore from", INCEPTION_v3_CKPT_PATH
      saver.restore(self.sess, INCEPTION_v3_CKPT_PATH)
    else:
      pass

  def preprocess_image(self, image):
    image = image.astype(np.float32)
    if NET != "inception":
      # change channel order BGR to RGB (by default opencv load image as BGR)
      image = image[:, :, [2, 1, 0]]
      image -= [_R_MEAN, _G_MEAN, _B_MEAN]
    else:
      image /= 255
      image -= 0.5
      image *= 2
    return image

  def extract_feature(self, image):
    if self.sess is None:
      self.sess = tf.Session(config=get_default_sess_config())
      self.build_extract_graph()

    assert len(image.shape) == 3
    image = self.preprocess_image(image)
    image = np.expand_dims(image, 0)

    ret = self.sess.run(self.extract_feature_op,
        feed_dict={self.image_placeholder: image})

    return np.squeeze(ret[0])


if __name__ == '__main__':
  ut = image_utils()

  img = cv2.imread("/home/fan/VOC/VOCdevkit/VOC2007/JPEGImages/003322.jpg")
  short_len = np.min(img.shape[0:2])

  img = cv2.resize(img, (299, 299))

  print img.shape
  feature = ut.extract_feature(img)
  print(feature.shape)
  print(feature)
  print(feature.dtype)
  cnt = 0
  for i in feature.reshape([-1]):
    if i != 0.:
      cnt += 1
  print cnt

  #feature = np.array(extract_feature(img))
  #feature2 = np.array(extract_feature([img, img, img]))
  #print feature
  #print feature.shape
  #print feature2
  #print feature2.shape
  # croped = crop_image(img, [0, 0, 200, 400], expand=False)
  # cv2.imshow("origin", img)
  # cv2.imshow("croped", croped)
  # cv2.waitKey(0)
