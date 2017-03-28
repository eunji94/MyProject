""" Deep Residual Network.

Applying a Deep Residual Network to BSDS500 Dataset.

"""

import tensorflow as tf
import numpy as np
import bsds500
import input_bsds500
import matplotlib.pyplot as plt

import sys
import numpy
from scipy import signal
from scipy import ndimage

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = numpy.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = numpy.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def block(X, W1, W2):
    conv1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    relu1 = tf.nn.relu(conv1)
    conv2 = tf.nn.conv2d(X, W2, strides=[1,1,1,1], padding='SAME')
    relu2 = tf.nn.relu(conv2)

    return X+relu2

def ssim(img1, img2, cs_map=False):
    """
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))
    """
    return 0

def psnr(target, ref):
	diff = ref - target
	diff = tf.reshape(diff, [tf.size(diff)])
	rmse = tf.sqrt( tf.reduce_mean(diff ** 2.) )
	return 20*tf.log(1.0/rmse)/tf.log(tf.constant(10.))

total_step = 100
batch_size = 32
width = 50
height = 50
n = 10

# Data loading
(trainX, trainY), (testX, testY), (valX, valY) = input_bsds500.load_data()
trainX = trainX.reshape([-1, 50, 50, 3])
trainY = trainY.reshape([-1, 50, 50, 3])
testX = testX.reshape([-1, 50, 50, 3])
textY = testY.reshape([-1, 50, 50, 3])
valX = valX.reshape([-1, 50, 50, 3])
valY = valY.reshape([-1, 50, 50, 3])
"""
path = 'X_train.txt'
new_file = open(path, 'r')
Xtrain = []
for line in new_file:
    Xtrain = np.append(Xtrain, float(line))
new_file.close()
Xtrain = Xtrain.reshape([-1, width, height, 3])
print (Xtrain.shape)

path = 'Y_train.txt'
new_file = open(path, 'r')
Ytrain = []
for line in new_file:
    Ytrain = np.append(Ytrain, float(line))
new_file.close()
Ytrain = Ytrain.reshape([-1, width, height, 3])
"""

# Building Residual Network
X = tf.placeholder("float", [None, width, height, 3])
Y = tf.placeholder("float", [None, width, height, 3])

w1 = init_weights([3, 3, 3, 64])
w2 = init_weights([n, 3, 3, 64, 64])
w3 = init_weights([n, 3, 3, 64, 64])
w4 = init_weights([width, height, 64, 3])

net = tf.nn.relu(tf.nn.conv2d(X, w1, strides=[1,1,1,1], padding='SAME'))
for i in range(n):
    net = block(net, w2[i], w3[i])
Y_ = tf.nn.conv2d(net, w4, strides=[1,1,1,1], padding='SAME')

cost = tf.reduce_mean(tf.squared_difference(Y_, Y))
eval_psnr = psnr(Y_, Y)
eval_ssim = ssim(Y_, Y)
train_op = tf.train.AdamOptimizer(0.001, 0.9).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(total_step):
        avg_cost = 0.
        avg_psnr = 0.
        avg_ssim = 0.
        training_batch = zip(range(0, len(trainX), batch_size),
                             range(batch_size, len(trainX)+1, batch_size))
        for start, end in training_batch:
            _, c = sess.run([train_op, cost],
                           feed_dict={X: trainX[start:end],
                                      Y: trainY[start:end]})

            avg_cost += c/training_batch
        if (step+1) % 4 == 0:
            eval_batch = zip(range(0, len(valX), batch_size),
                             range(batch_size, len(valX)+1, batch_size))
            for start, end in eval_batch:
                p, s = sess.run([train_op, eval_psnr, eval_ssim],
                                feed_dict={X: valX[start:end], Y: valY[start:end]})
                avg_psnr += p/eval_batch
                avg_ssim += s/eval_batch

        if (step+1) % 10 == 0:
            print("Step: ", '%4d'%(step+1), "Cost = ", "{:.9f}".format(avg_cost),
                  "PSNR =", "{:.9f}".format(avg_psnr), "SSIM =", "{:.9f}".format(avg_ssim))

    print("Optimization Finished!")
    
    import random
    r = random.randrange(len(testX))
    prediction = sess.run(Y_, {X: testX[r:r+1]})
    a = fig.add_subplot(1,3,1)
    a.set_title('Noise Image(Input)')
    plt.imshow(testX[r:r+1])
    b = fig.add_subplot(1,3,2)
    b.set_title('Denoise Image(Output)')
    b.set_xlabel("PSNR = ", "{:.9f}".format(psnr(prediction, testY[r:r+1])),
                 "SSIM = ", "{:.9f}".format(ssim(prediction, testY[r:r+1])))
    plt.imshow(prediction)
    c = fig.add_subplot(1,3,3)
    c.set_title('Clean Image(Compare)')
    plt.imshow(testY[r:r+1])

    fig.suptitle('Random Test')
    plt.show()
    
"""
filename = "sample.jpeg"
image = tf.image.decode_jpeg(filename, channels = 3)
float_images = tf.cast(image, tf.float32)
tf.image_summary('image', float_images)
"""
