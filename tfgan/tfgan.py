import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan

def _preprocess(element):
    # TODO: Implement
    pass

def _dense(inputs, units, l):
    return tf.layers.dense(
      inputs, units, None,
      kernel_initializer=tf.keras.initializers.glorot_uniform,
      kernel_regularizer=tf.keras.regularizers.l2(l=l),
      bias_regularizer=tf.keras.regularizers.l2(l=l))

def _batch_norm(inputs, mode):
    return tf.layers.batch_normalization(
      inputs, momentum=0.999, epsilon=0.001, training=mode)

def _deconv2d(inputs, filters, kernel_size, stride, l):
  return tf.layers.conv2d_transpose(
      inputs, filters, [kernel_size, kernel_size], strides=[stride, stride], 
      activation=tf.nn.relu, padding='same',
      kernel_initializer=tf.keras.initializers.glorot_uniform,
      kernel_regularizer=tf.keras.regularizers.l2(l=l),
      bias_regularizer=tf.keras.regularizers.l2(l=l))

def _conv2d(inputs, filters, kernel_size, stride, l):
  return tf.layers.conv2d(
      inputs, filters, [kernel_size, kernel_size], strides=[stride, stride], 
      activation=None, padding='same',
      kernel_initializer=tf.keras.initializers.glorot_uniform,
      kernel_regularizer=tf.keras.regularizers.l2(l=l),
      bias_regularizer=tf.keras.regularizers.l2(l=l))

def _generator(noise, mode, decay = 2.5e-5):
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    net = _dense(noise, 1024, decay)
    net = _dense(noise, 1024, decay)
    net = _batch_norm(net, training)
    net = tf.nn.relu(net)
    
    net = _dense(net, 7 * 7 * 256, training)
    net = _batch_norm(net, training)
    net = tf.nn.relu(net)
    
    net = tf.reshape(net, [-1, 7, 7, 256])
    net = _deconv2d(net, 64, 4, 2, decay)
    net = _deconv2d(net, 64, 4, 2, decay)
    
    net = _conv2d(net, 1, 4, 1, 0.0)
    net = tf.tanh(net)

    return net

def _discriminator(img, net, mode, decay = 2.5e-5):
    lrelu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)
    training = (mode == tf.estimator.ModeKeys.TRAIN)
  
    net = _conv2d(img, 64, 4, 2, decay)
    net = lrelu(net)
    
    net = _conv2d(net, 128, 4, 2, decay)
    net = lrelu(net)
    
    net = tf.layers.flatten(net)
    
    net = _dense(net, 1024, decay)
    net = _batch_norm(net, training)
    net = lrelu(net)
    
    net = _dense(net, 1, decay)

    return net