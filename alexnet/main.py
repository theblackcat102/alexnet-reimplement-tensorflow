import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

from alexnet.alexnet import AlexNet
from alexnet.utils import Dataset


def train(epochs, batch_size, learning_rate, dropout_rate):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train = Dataset(
        X=x_train.reshape(-1, 32, 32, 3) / 255,
        y=y_train.flatten(), batch_size=batch_size)
    test = Dataset(
        X=x_test.reshape(-1, 32, 32, 3) / 255,
        y=y_test.flatten(), batch_size=batch_size)

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    model = AlexNet(learning_rate=learning_rate)
    model.build()

    # train_writer = tf.summary.FileWriter('./log/train', model.sess.graph)
    # test_writer = tf.summary.FileWriter('./log/test')
    # saver = tf.train.Saver()
    step_counter = 0
    for epoch in range(epochs):
        with tqdm(total=len(train), ncols=120) as pbar:
            for i, (x_batch, y_batch) in enumerate(datagen.flow(train.X, train.y, batch_size=batch_size)):
                if i >= len(train):
                    break
                step_counter += 1
                loss, output = model.train(feed_dict={
                    model.x: x_batch,
                    model.y: y_batch,
                    model.dropout_rate: dropout_rate,
                })
                train_acc = accuracy_score(y_batch, output)
                pbar.set_description('loss: %.4f, train_acc: %.4f' % (
                    loss, train_acc))
                pbar.update(1)

                # summary = tf.Summary()
                # summary.value.add(tag="loss", simple_value=loss)
                # train_writer.add_summary(summary, step_counter)
        
        y_pred = []
        y_true = []
        total_loss = 0
        for x_batch, y_batch in train:
            output, loss = model.predict(x_batch, y_batch)
            y_pred.append(output.flatten())
            y_true.append(y_batch)
            total_loss += loss * len(x_batch)
        train_acc = accuracy_score(np.concatenate(y_true), np.concatenate(y_pred))
        total_loss /= len(train.X)

        y_pred = []
        y_true = []
        for x_batch, y_batch in test:
            output, _ = model.predict(x_batch, y_batch)
            y_pred.append(output.flatten())
            y_true.append(y_batch)
        test_acc = accuracy_score(np.concatenate(y_true), np.concatenate(y_pred))

        print('Epoch %2d/%2d, loss: %.4f, train_acc: %.4f, test_acc: %.4f' % (
            epoch + 1, epochs, total_loss, train_acc, test_acc))

    #     summary = tf.Summary()
    #     summary.value.add(tag="accuracy", simple_value=train_acc)
    #     train_writer.add_summary(summary, step_counter)

    #     summary = tf.Summary()
    #     summary.value.add(tag="accuracy", simple_value=test_acc)
    #     test_writer.add_summary(summary, step_counter)
    # saver.save(model.sess, save_path='log/model')

if __name__ == "__main__":
    train(
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        dropout_rate=0.5)