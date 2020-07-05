import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import cv2
import math
from timeit import default_timer as timer


tf.set_random_seed(1)

def print_ops():
    print(">>>"*10, "Graph Operations", "<<<"*10)
    for op in tf.get_default_graph().get_operations():
        print(op.name)
    print("="*20)

# [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

def get_global_vars():
    global_vars = tf.global_variables()
    return global_vars

def get_trainable_vars():
    ## Moving mean & moving variance won't be found in trainable_variables
    trainable_vars = tf.trainable_variables()
    return trainable_vars

def print_list(l, name=""):
    print(">>>"*10, name, "<<<"*10)
    [print(var) for var in l]
    print("="*20)

# def 
# bn_vars = [v for v in tf.trainable_variables() if "batch_norm" in v.name]
# print(bn_vars)


def get_lena():
    #https://homepages.cae.wisc.edu/~ece533/images/lena.png
    lena = cv2.imread("dataset/train_input/lena.png")
    lena = cv2.resize(lena, (512, 512))
    lena = np.reshape(lena, (1, 512, 512, 3))
    lena = lena / 255.0  #Not normalizing so that we can see lena
    return lena

def read_img(file):
    img = cv2.imread(file)
    img = cv2.resize(img, (512,512))
    img = img.astype(np.float32) / 255.0
    # img = np.reshape(img, (None,512,512,3))
    return img

def get_tr_dataset():
    train_input=[]
    train_label=[]
    for file_name in os.listdir("dataset/train_input"):
        if(file_name[-3:]=='png'):
            train_input.append(read_img("dataset/train_input/"+file_name))
            train_label.append(read_img("dataset/train_label/"+file_name))
    return train_input, train_label

def get_te_dataset():
    test_input=[]
    for file_name in os.listdir("dataset/test_input"):
        if(file_name[-3:]=='png'):
            test_input.append(read_img("dataset/test_input/"+file_name))
    return test_input

def save_model_out(input_float, out_name="model_out"):
    input_float = np.reshape(input_float, (512, 512, 3))
    img = input_float #/ input_float.max()
    img = img*255
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    print(img.max())
    print(img.min())
    cv2.imwrite(out_name+".png", img)
    print("Image saved!!", out_name+".png")


'''
===Notes===
loss = mse #1 - 10*(tf.math.log((255**2/mse)) / tf.math.log(10.0) ) #PSNR
# loss = tf.losses.absolute_difference(output, label_placeholder)
'''