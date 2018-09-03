from __future__ import division
from keras.layers import Conv2D,UpSampling2D
from keras.losses import binary_crossentropy
from keras.activations import sigmoid
from keras.layers.advanced_activations import LeakyReLU
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Lambda, add
from keras.models import Model
import keras.backend as k
import tensorflow as tf
from keras.initializers import RandomNormal



def conv2d(c_i, filters, ks=4,s=2, padding='SAME', activation=None):
    c = Conv2D(
        filters,
        kernel_size=ks,
        strides=s,
        padding=padding,
        activation=activation,
        kernel_initializer= RandomNormal(mean = 0.0, stddev= 0.02))(c_i)
    return c

def trans_conv2d(tc_i, filters, ks=4,s=2, activation=None, padding='SAME'):
    #t1 = Conv2DTranspose(filters, kernel_size=ks, strides=s, padding=padding,
    #                             activation=activation, kernel_initializer= RandomNormal(mean = 0.0, stddev= 0.02))(tc_i)
    t = UpSampling2D(size=4)(tc_i)
    t = Lambda(lambda x: tf.pad(x, [[0,0],[1,1],[1,1],[0,0]],'REFLECT'))(t)
    t1 = Conv2D(
        filters,
        kernel_size=ks,
        strides=s,
        padding=padding,
        activation=activation,
        kernel_initializer= RandomNormal(mean = 0.0, stddev= 0.02))(t)

    return t1

def residule_block(r_i, layer_output, ks=3, s=1):
    r = Lambda(lambda x: tf.pad(x, [[0,0],[1,1],[1,1],[0,0]],'REFLECT'))(r_i)
    #r = ReflectionPadding2D(padding=(1,1))(r_i)
    r = conv2d(r,layer_output,ks,s,padding= 'VALID')
    r = InstanceNormalization()(r)
    r = Lambda(lambda x: tf.pad(x, [[0,0],[1,1],[1,1],[0,0]],'REFLECT'))(r)
    #r = ReflectionPadding2D(padding=(1,1))(r)
    r = conv2d(r,layer_output,ks,s,padding= 'VALID')
    r = InstanceNormalization()(r)
    return add([r_i , r])

def discriminator(opt):
    img = Input(shape=(opt.data_pix_size,opt.data_pix_size,opt.in_dim,))
    d1 = LeakyReLU(alpha=0.2)(conv2d(img, opt.d_fir_dim, 4, 2))
    d2 = LeakyReLU(alpha=0.2)(InstanceNormalization()(conv2d(d1, opt.d_fir_dim*2, 4, 2)))
    d3 = LeakyReLU(alpha=0.2)(InstanceNormalization()(conv2d(d2, opt.d_fir_dim*4, 4, 2)))
    d4 = LeakyReLU(alpha=0.2)(InstanceNormalization()(conv2d(d3, opt.d_fir_dim*8, 4, 2)))
    d5 = conv2d(d4, 1, s=1)
    return Model(inputs = img, outputs = d5)


def generator_resnet(opt):
    img = Input(shape=(opt.data_pix_size,opt.data_pix_size,opt.in_dim,))
    pad_img = Lambda(lambda x: tf.pad(x, [[0,0],[3,3],[3,3],[0,0]],'REFLECT'))(img)
    c1 = conv2d(pad_img, opt.g_fir_dim, 7, 1,padding='VALID',activation='relu')
    c1 = InstanceNormalization()(c1)
    c2 = conv2d(c1, opt.g_fir_dim*2, 3, 2,activation='relu')
    c2 = InstanceNormalization()(c2)
    c3 = conv2d(c2, opt.g_fir_dim*4, 3, 2,activation='relu')
    c3 = InstanceNormalization()(c3)
    #residule bolck
    r1 = residule_block(c3, opt.g_fir_dim*4)
    r2 = residule_block(r1, opt.g_fir_dim*4)
    r3 = residule_block(r2, opt.g_fir_dim*4)
    r4 = residule_block(r3, opt.g_fir_dim*4)
    r5 = residule_block(r4, opt.g_fir_dim*4)
    r6 = residule_block(r5, opt.g_fir_dim*4)
    r7 = residule_block(r6, opt.g_fir_dim*4)
    r8 = residule_block(r7, opt.g_fir_dim*4)
    r9 = residule_block(r8, opt.g_fir_dim*4)

    t1 = trans_conv2d(r9, opt.g_fir_dim*2, 3, 2,padding='SAME',activation='relu')
    t1 = InstanceNormalization()(t1)
    t2 = trans_conv2d(t1, opt.g_fir_dim, 3, 2,padding='SAME',activation='relu')
    t2 = InstanceNormalization()(t2)
    t2_pad = Lambda(lambda x: tf.pad(x, [[0,0],[1,2],[1,2],[0,0]],'REFLECT'))(t2)
    gen_img = conv2d(t2_pad, opt.out_dim, 7, 1, padding='VALID',activation='tanh')
    return Model(inputs = img, outputs =gen_img)


def abs_criterion(x,y):
    return k.mean(k.abs(x-y))

def mae_criterion(x,y):
    return k.mean((x-y)**2)

def sce_criterion(logit, label):
    return k.mean(sigmoid(binary_crossentropy(label,logit)))