import os, glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from matplotlib.pyplot import figure
from tensorflow.python.platform import flags
import argparse


parser = argparse.ArgumentParser(description='AlexNet first layer visualization')
parser.add_argument('--model', '-m',type=str, help='model checkpoint')
args = parser.parse_args()

target = args.model

reader = pywrap_tensorflow.NewCheckpointReader(target)
var_to_shape_map = reader.get_variable_to_shape_map()
conv1_weights = reader.get_tensor('conv2d/kernel')
filter_size = conv1_weights.shape[0]
kernel_padding = 1
visualize_size = filter_size+kernel_padding*2
all_kernel = np.zeros((6*visualize_size, 16*visualize_size, 3 ))

for j in range(6):
    for i in range(16):
        idx = j*6 + i
        norm = conv1_weights[:,:,:, idx]
        norm = (norm - norm.min())/(norm.max() - norm.min())
#         print(all_kernel[j*17+3:(j+1)*17-3,i*17+3:(i+1)*17-3,  :].shape)
        all_kernel[j*visualize_size+kernel_padding:(j+1)*visualize_size-kernel_padding,
                   i*visualize_size+kernel_padding:(i+1)*visualize_size-kernel_padding,
                   :] = norm
figure(num=None, figsize=(11, 8), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(all_kernel)
plt.show()


