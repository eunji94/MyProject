"""BSDS500 Dataset


"""

from __future__ import absolute_import, print_function


import tensorflow as tf
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

width = 200
height = 200

def load_data(dirname="BSR/BSDS500/data/images"):

    print("Start Data Loading")
    fpath = os.path.join(dirname, 'train')
    X_train, Y_train = jpg_to_tensor(fpath)
    print("Finish Train Data Loading")
    print("Train Data: ", len(X_train))
    
    fpath = os.path.join(dirname, 'test')
    X_test, Y_test = jpg_to_tensor(fpath)
    print("Finish Test Data Loading")
    print("Test Data: ", len(X_test))
    
    fpath = os.path.join(dirname, 'val')
    X_val, Y_val = jpg_to_tensor(fpath)
    print("Finish Validation Data Loading")
    print("Validation Data: ", len(X_val))
    
    return (X_train, Y_train), (X_test, Y_test), (X_val, Y_val)
    

def jpg_to_tensor(dirname):
    fpath = os.path.join(dirname,'*.jpg')
    file_list = glob.glob(fpath)
    X = []
    Y = []
    for i in range(len(file_list)):
        jpeg_r = tf.read_file(file_list[i])
        image = tf.image.decode_jpeg(jpeg_r, channels=3)
        resized_image = tf.image.resize_image_with_crop_or_pad(image, width, height)
        gaussian_image = Gaussian_noise_layer(resized_image, 10)
        gaussian_image = tf.cast(gaussian_image, tf.float16)
        resized_image = tf.cast(resized_image, tf.float16)
        
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        tf.train.start_queue_runners(sess)
        y = sess.run(resized_image)
        x = sess.run(gaussian_image)
        """
        fig = plt.figure()
        fig.add_subplot(1,2,1)
        plt.imshow(y)
        fig.add_subplot(1,2,2)
        plt.imshow(x)
        plt.show()
        """
        if i == 0:
            X = x
            Y = y
        elif i == 1:
            X = np.stack((X,x))
            Y = np.stack((Y,y))
        else:
            X = np.concatenate((X, x[None, ...]))
            Y = np.concatenate((Y, y[None, ...]))
        if (i+1) % 20 == 0:
            print(i+1, " / ", len(file_list), "Loading...")

    return X, Y

def Gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape = input_layer.get_shape(), mean = 0.0, stddev = std, dtype = tf.float32)
    input_layer = tf.to_float(input_layer)
    return input_layer + noise

def datasaver():
    (X_train, Y_train), (X_test, Y_test), (X_val, Y_val) = load_data()
    path = 'X_train.txt'
    X_train = X_train.reshape((len(X_train)*width*height*3))
    data_file = open(path, 'w')
    for item in X_train:
        data_file.write("%s\n" % item)
    data_file.close()
    
    path = 'Y_train.txt'
    Y_train = Y_train.reshape((len(Y_train)*width*height*3))
    data_file = open(path, 'w')
    for item in Y_train:
        data_file.write("%s\n" % item)
    data_file.close()

    path = 'X_test.txt'
    X_test = X_test.reshape((len(X_test)*width*height*3))
    data_file = open(path, 'w')
    for item in X_test:
        data_file.write("%s\n" % item)
    data_file.close()

    path = 'Y_test.txt'
    Y_test = Y_test.reshape((len(Y_test)*width*height*3))
    data_file = open(path, 'w')
    for item in Y_test:
        data_file.write("%s\n" % item)
    data_file.close()

    path = 'X_val.txt'
    X_val = X_val.reshape((len(X_val)*width*height*3))
    data_file = open(path, 'w')
    for item in X_train:
        data_file.write("%s\n" % item)
    data_file.close()

    path = 'Y_val.txt'
    Y_val = Y_val.reshape((len(Y_val)*width*height*3))
    data_file = open(path, 'w')
    for item in X_train:
        data_file.write("%s\n" % item)
    data_file.close()
