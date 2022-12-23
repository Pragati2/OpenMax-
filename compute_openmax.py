import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp
from scipy.io import loadmat

from openmax_utils import *
from evt_fitting import weibull_distribution_fitting, query_weibul_distribution

try:
    import libmr
except ImportError:
    print ("LibMR not installed or libmr.so not found")
    print ("Install libmr: cd libMR/; ./compile.sh")
    sys.exit()


#---------------------------------------------------------------------------------
# params and configuratoins
NumChannels = 1
NumClasses = 10
ALPHA_RANK = 6
WEIBULL_TAIL_SIZE = 10

#---------------------------------------------------------------------------------
def computeOpenMaxProbability(openmaxFC8, openmaxScore):
    """ Convert the scores in probability value using openmax
    
    Input:
    ---------------
    openmaxFC8 : modified FC8 layer from Weibull based computation
    openmaxScore : degree

    Output:
    ---------------
    modifiedScores : probability values modified using OpenMax framework,
    by incorporating degree of uncertainity/openness for a given class
    
    """
    probScores, probUnknowns = [], []
    for channel in range(NumChannels):
        channelScores, channel_unknowns = [], []
        for category in range(NumClasses):
            #print (channel,category)
            #print ('openmax',openmaxFC8[channel, category])

            channelScores += [sp.exp(openmaxFC8[channel, category])]
        #print ('CS',channelScores)

        TOTAL_DENOM = sp.sum(sp.exp(openmaxFC8[channel, :])) + sp.exp(sp.sum(openmaxScore[channel, :]))
        #print (TOTAL_DENOM)

        probScores += [channelScores/TOTAL_DENOM ]
        #print (probScores)

        probUnknowns += [sp.exp(sp.sum(openmaxScore[channel, :]))/TOTAL_DENOM]
        
    probScores = sp.asarray(probScores)
    probUnknowns = sp.asarray(probUnknowns)

    scores = sp.mean(probScores, axis = 0)
    unknowns = sp.mean(probUnknowns, axis=0)
    modifiedScores =  scores.tolist() + [unknowns]
    assert len(modifiedScores) == 11
    return modifiedScores

#---------------------------------------------------------------------------------
def recalibrate_scores(weibullDistributionModel, labellist, img_arr,
                       layer = 'fc8', alpharank = 6, distance_type = 'eucos'):
    """ 
    Given FC8 features for an image, list of weibull models for each class,
    re-calibrate scores

    Input:
    ---------------
    weibullDistributionModel : pre-computed weibullDistributionModel obtained from weibull_distribution_fitting() function
    labellist : ImageNet 2012 labellist
    img_arr : features for a particular image extracted using caffe architecture
    
    Output:
    ---------------
    openmax_probab: Probability values for a given class computed using OpenMax
    softmax_probab: Probability values for a given class computed using SoftMax (these
    were precomputed from caffe architecture. Function returns them for the sake 
    of convienence)

    """
    
    imglayer = img_arr[layer]
    ranked_list = img_arr['scores'].argsort().ravel()[::-1]
    alpha_weights = [((alpharank+1) - i)/float(alpharank) for i in range(1, alpharank+1)]
    ranked_alpha = sp.zeros(10)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    #print (imglayer)
    # Now recalibrate each fc8 score for each channel and for each class
    # to include probability of unknown
    openmaxFC8, openmaxScore = [], []
    for channel in range(NumChannels):
        channelScores = imglayer[channel, :]
        opnemaxFC8_channel = []
        openmaxFC8_unknown = []
        count = 0
        for categoryid in range(NumClasses):
            # get distance between current channel and mean vector
            categoryWeibull = query_weibul_distribution(labellist[categoryid], weibullDistributionModel, distance_type = distance_type)

            #print (categoryWeibull[0],categoryWeibull[1],categoryWeibull[2])

            channel_distance = compute_distance(channelScores, channel, categoryWeibull[0],
                                                distance_type = distance_type)
            #print ('cd',channel_distance)                                                
            # obtain w_score for the distance and compute probability of the distance
            # being unknown wrt to mean training vector and channel distances for
            # category and channel under consideration
            wscore = categoryWeibull[2][channel].w_score(channel_distance)
            #print ('wscore',wscore)
            #print (channelScores)
            modified_fc8_score = channelScores[categoryid] * ( 1 - wscore*ranked_alpha[categoryid] )
            opnemaxFC8_channel += [modified_fc8_score]
            openmaxFC8_unknown += [channelScores[categoryid] - modified_fc8_score ]

        # gather modified scores fc8 scores for each channel for the given image
        openmaxFC8 += [opnemaxFC8_channel]
        openmaxScore += [openmaxFC8_unknown]
    openmaxFC8 = sp.asarray(openmaxFC8)
    openmaxScore = sp.asarray(openmaxScore)
    
    #print (openmaxFC8,openmaxScore)
    # Pass the recalibrated fc8 scores for the image into openmax    
    openmax_probab = computeOpenMaxProbability(openmaxFC8, openmaxScore)
    softmax_probab = img_arr['scores'].ravel() 
    return sp.asarray(openmax_probab), sp.asarray(softmax_probab)

#---------------------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()


    # Optional arguments.
    parser.add_argument(
        "--weibull_tailsize",
        type=int,
        default=WEIBULL_TAIL_SIZE,
        help="Tail size used for weibull fitting"
    )
    
    parser.add_argument(
        "--alpha_rank",
        type=int,
        default=ALPHA_RANK,
        help="Alpha rank to be used as a weight multiplier for top K scores"
    )

    parser.add_argument(
        "--distance",
        default='eucos',
        help="Type of distance to be used for calculating distance \
        between mean vector and query image \
        (eucos, cosine, euclidean)"
    )

    parser.add_argument(
        "--mean_files_path",
        default='data/mean_files/',
        help="Path to directory where mean activation vector (MAV) is saved."        
    )

    parser.add_argument(
        "--synsetfname",
        default='synset_words_caffe_ILSVRC12.txt',
        help="Path to Synset filename from caffe website"        
    )

    parser.add_argument(
        "--image_arrname",
        default='data/train_features/n01440764/n01440764_14280.JPEG.mat',
        help="Image Array name for which openmax scores are to be computed"        
    )

    parser.add_argument(
        "--distance_path",
        default='data/mean_distance_files/',
        help="Path to directory where distances of training data \
        from Mean Activation Vector is saved"        
    )

    args = parser.parse_args()

    distance_path = args.distance_path
    mean_path = args.mean_files_path
    alpha_rank = args.alpha_rank
    weibull_tailsize = args.weibull_tailsize
    synsetfname = args.synsetfname
    image_arrname = args.image_arrname

    labellist = getlabellist(synsetfname)
    weibullDistributionModel = weibull_distribution_fitting(mean_path, distance_path, labellist,
                                        tailsize = WEIBULL_TAIL_SIZE)

    print ("Completed Weibull fitting on %s models" %len(weibullDistributionModel.keys()))
    img_arr = loadmat(image_arrname)
    openmax, softmax =  recalibrate_scores(weibullDistributionModel, labellist, img_arr)
    print ("Image ArrName: %s" %image_arrname)
    print ("Softmax Scores ", softmax)
    print ("Openmax Scores ", openmax)
    print (openmax.shape, softmax.shape)


if __name__ == "__main__":
    main()
