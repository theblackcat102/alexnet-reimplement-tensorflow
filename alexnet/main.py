import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator

from alexnet.alexnet import AlexNet
from alexnet.utils import Dataset


def batch_gen(X, y, batch_size, shuffle=True):
    assert(X.shape[0] == y.shape[0])
    indices = np.arange(len(X))
    if shuffle:
        np.random.shuffle(indices)
    start_index = 0
    while start_index < len(X):
        index = indices[start_index: start_index + batch_size]
        X_batch = X[index]
        y_batch = y[index]
        start_index += batch_size
        yield X_batch, y_batch

def train(epochs, batch_size, learning_rate, dropout_rate):
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    train = Dataset(
        X=x_train.reshape(-1, 32, 32, 3),
        y=np.argmax(y_train, axis=-1), batch_size=batch_size)
    test = Dataset(
        X=x_test.reshape(-1, 32, 32, 3),
        y=np.argmax(y_test, axis=-1), batch_size=batch_size)

    # datagen = ImageDataGenerator(
    #     featurewise_center=True,
    #     featurewise_std_normalization=True,
    #     rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     horizontal_flip=True)
    # datagen.fit(x_train)

    model = AlexNet(learning_rate=learning_rate)

    # train_writer = tf.summary.FileWriter('./log/train', sess.graph)
    # test_writer = tf.summary.FileWriter('./log/test')
    for i in range(epochs):
        with tqdm(total=len(train), ncols=80) as pbar:
            for x_batch, y_batch in train:
                loss, output = model.train(feed_dict={
                    model.x: x_batch,
                    model.y: y_batch,
                    model.dropout: dropout_rate,
                })
                train_acc = accuracy_score(y_batch, output)
                pbar.set_description('loss: %.4f, train_acc: %.4f' % (
                    loss, train_acc))
                pbar.update(1)

                # train_summary = tf.Summary()
                # train_summary.value.add(tag="loss", simple_value=loss)
                # train_writer.add_summary(train_summary, step_counter)
        
            y_pred = []
            for x_batch, y_batch in test:
                output = model.predict({
                    model.x: x_batch,
                    model.y: y_batch,
                })
                y_pred.extend(output.flatten())
            test_acc = accuracy_score(test.y, np.concatenate(y_pred))

            pbar.set_description('loss: %.4f, train_acc: %.4f, test_acc: %.4f' % (
                loss, train_acc, test_acc))    

if __name__ == "__main__":
    train(
        epochs=50,
        batch_size=256,
        learning_rate=0.001,
        dropout_rate=0.5)