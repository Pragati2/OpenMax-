from __future__ import print_function
import keras
from keras.dataset import cifar10
from keras.models import Sequential,load_model,save_model,model_from_config,model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import plot_model
import tensorlayer as tl


from evt_fitting import weibull_tailfitting, query_weibull
from compute_openmax import computeOpenMaxProbability,recalibrate_scores
from openmax_utils import compute_distance

import scipy.spatial.distance as spd
import h5py

import libmr

import numpy as np
import scipy

import pickle
import matplotlib.pyplot as plt

import ssl
import sys
import cv2
import glob

ssl._create_default_https_context = ssl._create_unverified_context

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def seperate_data(x,y):
    ind = y.argsort()
    sort_x = x[ind[::-1]]
    sort_y = y[ind[::-1]]

    dataset_x = []
    dataset_y = []
    mark = 0

    for a in range(len(sort_y)-1):
        if sort_y[a] != sort_y[a+1]:
            dataset_x.append(np.array(sort_x[mark:a]))
            dataset_y.append(np.array(sort_y[mark:a]))
            mark = a
        if a == len(sort_y)-2:
            dataset_x.append(np.array(sort_x[mark:len(sort_y)]))
            dataset_y.append(np.array(sort_y[mark:len(sort_y)]))
    return dataset_x,dataset_y

def compute_feature(x,model):
    score = get_activations(model,8,x)
    fc8 = get_activations(model,7,x)
    return score,fc8

def compute_mean_vector(feature):
    return np.mean(feature,axis=0)

def compute_distances(mean_feature,feature,category_name):
    eucos_dist, eu_dist, cos_dist = [], [], []
    eu_dist,cos_dist,eucos_dist = [], [], []
    for feat in feature:
        eu_dist += [spd.euclidean(mean_feature, feat)]
        cos_dist += [spd.cosine(mean_feature, feat)]
        eucos_dist += [spd.euclidean(mean_feature, feat)/200. + spd.cosine(mean_feature, feat)]
    distances = {'eucos':eucos_dist,'cosine':cos_dist,'euclidean':eu_dist}
    return distances

# def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
#     with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
#         # note the encoding type is 'latin1'
#         batch = pickle.load(file, encoding='latin1')
#     features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
#     labels = batch['labels']
#     return features, labels

batch_size = 128
num_classes = 10
epochs = 50
data_augmentation = True
num_predictions = 20
save_dir = os.path.join("/home/pragati/OSDN_CIFAR", 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
img_rows, img_cols = 32, 32

if K.image_datat_format == 'channel first':
    x_train = x_train.reshape(x_train[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test[0], 1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test[0], img_rows, img_cols, 1)

x_train /= 255.0
x_test /= 255.0

