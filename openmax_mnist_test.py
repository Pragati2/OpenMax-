import numpy as np
# from tensorflow import keras
from openmax import *
import glob
import cv2
import os 

np.random.seed(12345)
model = load_model('/home/pragati-home/OSDN_CIFAR/saved_models/keras_cifar10_trained_model.h5')
create_model(model)

X_test_new = []
Y_test_new = []

path = glob.glob("/home/OSDN_CIFAR/images_MNIST/*.png")
for imagepath in path:
    n = cv2.imread(imagepath)
    n = n.reshape(n, 28,28,)
    X_test_new.append(n)
    
# X_test_new = X_test_new.reshape(X_test_new[0],28,28,1)
Y_test_new = ['1', '2', '3','0','unknown']
# Y_test_new = lambda X_test_new: X_test_new, (10, 0, 10, 10, 10)
# X_test_new = X_test_new.reshape(X_test_new[0],28,28,1)
# Y_test_new = Y_test_new.reshape(Y_test_new[0],28, 28, 1)
print(X_test_new)

for i in range(0,4):

    # random_char = np.random.randint(0,10000)
    test_x1 = X_test_new[i]
    test_y1 = Y_test_new[i]
    
    # test_x1 = test_x1.reshape(test_x1, 32,32,3)
    # test_y1.reshape(-1)

    image_show(test_x1, test_y1)

    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle = True, **k)


    # Compute fc8 activation for the given image

    activation = compute_activation(model, test_x1)
    #print (activation)


    # Compute openmax 

    softmax,openmax = compute_openmax(model,activation)
    #openmax_unknown_class(model)
    np.load = np_load_old

    print ('Actual Label: ', np.argmax(test_y1))
    print ('Prediction Softmax: ', softmax)
    if openmax == 10:
        openmax = 'Unknown'
    print ('Prediction openmax: ',openmax)
    i = i + 1
    # model  = load_model('MNIST_CNN_tanh.h5')
    # create_model(model)