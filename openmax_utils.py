import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp
import numpy as np

def parse_synsetfile(synsetfname):
    """ Read ImageNet 2012 file
    """
    category_list = open(synsetfname, 'r').readlines()
    image_net_IDs = {}
    count = 0
    for categoryinfo in category_list:
        wnetid = categoryinfo.split(' ')[0]
        category_name = ' '.join(categoryinfo.split(' ')[1:])
        image_net_IDs[str(count)] = [wnetid, category_name]
        count += 1

    assert len(image_net_IDs.keys()) == 1000
    return image_net_IDs

def getlabel_list(synsetfname):
    """ read sysnset file as python list. Index corresponds to the output that 
    caffe provides
    """
    
    category_list = open(synsetfname, 'r').readlines()
    label_list = [category.split(' ')[0] for category in category_list]
    return label_list


def computeDistance(query_channel, channel, meanVec, distance_type = 'eucos'):
    """ Compute the specified distance type between chanels of mean vector and query image.
    In caffe library, FC8 layer consists of 10 channels. Here, we compute distance
    of distance of each channel (from query image) with respective channel of
    Mean Activation Vector. In the paper, we considered a hybrid distance eucos which
    combines euclidean and cosine distance for bouding open space. Alternatively,
    other distances such as euclidean or cosine can also be used. 
    
    Input:
    *-*-*-*-*-*-*
    query_channel: Particular FC8 channel of query image
    channel: channel number under consideration
    meanVec: mean activation vector

    Output:
    --------
    query_distance : Distance between respective channels

    """
    #print ('copute',query_channel,channel,meanVec)
    #print ('mean',meanVec)
    #print ('query_ch',query_channel)
    #print ('channel',channel)

    query_channel = np.array(query_channel)
    meanVec = np.reshape(meanVec,(10,1))
    #print ('shape',meanVec.shape)
    #exit()
    #print (distance_type)
    if distance_type == 'eucos':
        #print (meanVec.shape,query_channel.shape)
        query_distance = spd.euclidean(meanVec, query_channel)/200. + spd.cosine(meanVec, query_channel)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(meanVec, query_channel)/200.
    elif distance_type == 'cosine':
        query_distance = spd.cosine(meanVec, query_channel)
    else:
        print ("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance
    
