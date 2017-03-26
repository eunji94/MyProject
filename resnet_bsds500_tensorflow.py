""" Deep Residual Network.

Applying a Deep Residual Network to BSDS500 Dataset.

"""

import tensorflow as tf
import numpy as np
import bsds500

def block(X, W1, W2):
    conv1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    relu1 = tf.nn.relu(conv1)
    conv2 = tf.nn.conv2d(X, W2, strides=[1,1,1,1], padding='SAME')
    relu2 = tf.nn.relu(conv2)

    return X+relu

def ssim(img1, img2, cs_map=False):
    size = 11
    sigma = 1.5
    window = gauss.fspecial_gauss(size, sigma)
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

def psnr(target, ref):
	diff = ref - target
	diff = diff.flatten('C')
	rmse = math.sqrt( np.mean(diff ** 2.) )
	return 20*math.log10(1.0/rmse)

total_step = 100
batch_size = 32
width = 50
height = 50
n = 10

# Data loading
(trainX, trainY), (testX, testY), (valX, valY) = load_data()
X = X.reshape([-1, 50, 50, 3])
Y = Y.reshape([-1, 50, 50, 3])
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

init = tf.initialize_all_variables()

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
                                feed_dict={X: evalX[start:end], Y: evalY[start:end]})
                avg_psnr += p/eval_batch
                avg_ssim += s/eval_batch

        if (step+1) % 10 == 0:
            print("Step: ", '%4d'%(step+1), "Cost = ", "{:.9f}".format(avg_cost),
                  "PSNR =", "{:.9f}".format(avg_psnr), "SSIM =", "{:.9f}".format(avg_ssim))

    
"""
filename = "sample.jpeg"
image = tf.image.decode_jpeg(filename, channels = 3)
float_images = tf.cast(image, tf.float32)
tf.image_summary('image', float_images)
"""
