""" Deep Residual Network.

Applying a Deep Residual Network to BSDS500 Dataset.

"""

from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow as tf
import bsds500

def resnet_block(incoming, nb_blocks, out_channels,
                 activation='relu', batch_norm=True,
                 bias=True, weights_init='variance_scaling',
                 bias_init='zeros', regularizer='L2', weight_decay=0.0001,
                 trainable=True, restore=True, reuse=False, scope=None,
                 name="ResnetBlock"):
    
    resnet = incoming

    # Variable Scope fix for older TF
    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:

        name = scope.name #TODO

        for i in range(nb_blocks):
            identity = resnet

            if batch_norm:
                resnet = tflearn.batch_normalization(resnet)
            resnet = tflearn.activation(resnet, activation='linear')

            resnet = tflearn.conv_2d(resnet, out_channels, 3, padding='same',
                             activation='linear', bias=True, weights_init='variance_scaling',
                             bias_init='zeros', regularizer='L2', weight_decay=0.0001,
                             trainable=True, restore=True)

            resnet = resnet + identity

    return resnet

def psnr(target, ref):
	diff = ref - target
	diff = diff.flatten('C')
	rmse = math.sqrt( np.mean(diff ** 2.) )
	return 20*math.log10(1.0/rmse)

# Residual blocks
n = 2

# Data loading
import input_bsds500
(X, Y), (testX, testY), (valX, valY) = input_bsds500.load_data()
X = X.reshape([-1, 50, 50, 3])
Y = Y.reshape([-1, 50, 50, 3])
testX = testX.reshape([-1, 50, 50, 3])
textY = testY.reshape([-1, 50, 50, 3])
valX = valX.reshape([-1, 50, 50, 3])
valY = valY.reshape([-1, 50, 50, 3])

"""
# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)
"""
"""
# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([50, 50], padding=1)
"""

# Building Residual Network
net = tflearn.input_data(shape=[None, 50, 50, 3])

net = tflearn.conv_2d(net, 64, 3, activation='relu')
# Residual blocks
net = resnet_block(net, n, 64)
net = resnet_block(net, n, 64)
net = resnet_block(net, n, 64)
net = resnet_block(net, n, 64)
net = resnet_block(net, n, 64)
# regression
net = tflearn.conv_2d(net, 3, 50, activation='relu')
opt = tflearn.Adam(learning_rate=0.001, beta1=0.9)
net = tflearn.regression(net, optimizer=opt, loss='mean_square')
# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet_bsds500',
                    max_checkpoints=10, tensorboard_verbose=0)

model.fit(X, Y, n_epoch=50, validation_set=(valX, valY),
          snapshot_epoch=False,
          show_metric=True, batch_size=32, shuffle=True,
          run_id='resnet_bsds500')
# Evaluation
for i in len(testX):
    eval_psnr = psnr(testY[i], model.predict(testX[i]))
    print('Test PSNR: %0.4f%%' % eval_psnr)

"""
filename = "sample.jpeg"
image = tf.image.decode_jpeg(filename, channels = 3)
float_images = tf.cast(image, tf.float32)
tf.image_summary('image', float_images)
"""
