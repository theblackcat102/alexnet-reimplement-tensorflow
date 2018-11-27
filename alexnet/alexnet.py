import tensorflow as tf
import numpy as np


class AlexNet(object):
    """Implementation of the AlexNet."""

    def __init__(
            self,
            learning_rate=0.001,
            input_shape=(32, 32, 3),
            num_classes=100,
            split=False):
        self.x = tf.placeholder(tf.float32, shape=[None, *input_shape])
        self.y = tf.placeholder(tf.int32, shape=[None])
        self.dropout_rate = tf.placeholder_with_default(0, shape=[])
        self.num_classes = num_classes
        
        # Convolution 1
        x = self.conv2d(self.x, 96, [3, 3], [1, 1], split=split)
        x = self.relu(x)
        x = self.lrn(x, radius=2, alpha=2e-5, beta=0.75)
        x = self.maxpool(x, [3, 3], [2, 2])
        
        # Convolution 2
        x = self.conv2d(x, 256, [3, 3], [1, 1], split=split)
        x = self.relu(x)
        x = self.lrn(x, radius=2, alpha=2e-5, beta=0.75)
        x = self.maxpool(x, [3, 3], [2, 2])

        # Convolution 3
        x = self.conv2d(x, 384, [3, 3], [1, 1], split=False)
        x = self.relu(x)

        # Convolution 4
        x = self.conv2d(x, 384, [3, 3], [1, 1], split=split)
        x = self.relu(x)

        # Convolution 5
        x = self.conv2d(x, 256, [3, 3], [1, 1], split=split)
        x = self.relu(x)
        x = self.maxpool(x, [3, 3], [2, 2])

        # Fully Connected 6
        x = tf.layers.Flatten()(x)
        x = self.dense(x, 4096)
        x = self.dropout(x, rate=self.dropout_rate)
        x = self.relu(x)

        # Fully Connected 7
        x = self.dense(x, 4096)
        x = self.dropout(x, rate=self.dropout_rate)
        x = self.relu(x)

        # Output
        x = self.dense(x, num_classes)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.one_hot(self.y, depth=num_classes), logits=x))
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.output = tf.argmax(x, axis=-1)

    def relu(self, x):
        return tf.nn.relu(x)

    def dense(self, x, units):
        x = tf.layers.dense(inputs=x, units=units)
        return x

    def maxpool(self, x, pool_size, strides, padding='same'):
        return tf.layers.max_pooling2d(
            inputs=x, pool_size=pool_size, strides=strides, padding=padding)

    def lrn(self, x, radius, alpha, beta, bias=1.0):
        """Create a local response normalization layer."""
        return tf.nn.local_response_normalization(
            x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

    def dropout(self, x, rate):
        x = tf.layers.dropout(x, rate=rate)
        return x

    def conv2d(self, x, filters, kernel_size, strides, padding='same',
               split=False):
        if split:
            x_up, x_down = tf.split(value=x, axis=-1, num_or_size_splits=2)
            x_up = self.conv2d(x_up, filters // 2, kernel_size, strides, padding)
            x_down = self.conv2d(x_down, filters // 2, kernel_size, strides, padding)
            x = tf.concat(values=[x_up, x_down], axis=-1)
        else:
            x = tf.layers.conv2d(
                inputs=x,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding)
        return x

    def build(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, feed_dict):
        fetches = [
            self.train_op,
            self.output,
            self.loss,
        ]
        _, output, loss = self.sess.run(fetches, feed_dict=feed_dict)
        return loss, output
    
    def predict(self, x):
        fetches = [
            self.output,
        ]
        output = self.sess.run(fetches, feed_dict={self.x: x})
        return output
