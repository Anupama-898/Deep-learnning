import random
from keras.layers.core import *
import sys
import cv2
import _pickle as cPickle
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import numpy.random as rng
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
from keras.models import Model, Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Input, concatenate, Add, GlobalAveragePooling2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D,Convolution2D
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.models import model_from_json,model_from_config,load_model
from keras.optimizers import SGD,RMSprop,adam,Adam
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.preprocessing import image
from keras import backend as K
from keras.initializers import random_uniform, RandomNormal
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from collections import OrderedDict as od
import tensorflow as tf
import json

def identity(e):
  skip_conn =tf.identity(e, name='identity')
  return skip_conn

def adaptive_global_average_pool_2d(x):
    c = x.get_shape()[-1]
    return tf.reshape(tf.reduce_mean(x, axis=[1, 2]), (-1, 1, 1, c))
	
	
def channel_attention(input_layer_2):
	skip_conn = Lambda(identity)(input_layer_2)
	x = Lambda(adaptive_global_average_pool_2d)(x)
	x = GlobalAveragePooling2D()(input_layer_2)
	x = Conv2D(64 // 16, (1, 1), activation='relu', padding='same',use_bias=True)(x)
	x = Conv2D(64, (1, 1), activation='sigmoid', padding='same',use_bias=True)(x)
	y =  Multiply()([x, skip_conn_1])
	return y
	
def residual_channel_attention_block(input_layer_1):
	skip_conn = Lambda(identity)(input_layer_1)
	x = Conv2D(64, (3, 3), activation='relu', padding='same',use_bias=True)(input_layer_1)
	x = Conv2D(64, (3, 3), padding='same',use_bias=True)(x)
	x = channel_attention(x)
	y = Add()([x, skip_conn])
  return y


def residual_group(input_layer):
  skip_conn = Lambda(identity)(input_layer)
  for i in range(20):
    x = residual_channel_attention_block(input_layer)
    x = Conv2D(64, (3, 3), padding='same',use_bias=True)(x)
	y = Add()([x, skip_conn])
  return y
	
def residual_channel_attention_network(input_dim):
	input_img = Input(shape=input_dim)
	head = Conv2D(64,(3,3),padding='same',use_bias=True)(input_img)
	x = head
	for i in range(10):
		x = residual_group(x)
	body = Conv2D(64,(3,3),padding='same',use_bias=True)(x)
	body = Add()([body, head]
	  #x = up_scaling(body, f, scale, name='up-scaling')
	tail = Conv2D(1,(3,3),padding='same',use_bias=True)(body)
	output_model = Model(input_img, tail)
	return output_model

	
input_shape = (144,144, 1)
model = residual_channel_attention_network(input_shape)
model.summary()