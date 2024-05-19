import sys
from collections import defaultdict

sys.path.append("../")

import numpy as np
from keras.layers import *
from Library.util import *
import keras
from keras.datasets import cifar10, mnist
import tensorflow as tf
from sklearn.utils import shuffle
import pandas as pd

# Define the TensorFlow data augmentation and normalization pipeline
def transform_train(image, label):
    # Random crop
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.random_crop(image, size=[50000, 32, 32, 3])

    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Normalize
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    image = (image - mean) / std

    return image, label

def transform_test(image, label):
    # Convert to tensor
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    # Normalize
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    image = (image - mean) / std

    return image, label


def load_dataset(data):
    # todo: Dataset Cifar10에 대해 preprocessing 추가.
    if data == 'mnist':
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        x_train = np.expand_dims(x_train, axis=-1)
        x_train = np.repeat(x_train, 3, axis=-1)
        x_train = tf.image.resize(x_train, [32, 32])  # if we want to resize

        x_test = np.expand_dims(x_test, axis=-1)
        x_test = np.repeat(x_test, 3, axis=-1)
        x_test = tf.image.resize(x_test, [32, 32])  # if we want to resize

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
    elif data == 'cifar10':
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)

def split_data(mode, origin_data, n_user, alpha):
    if mode == 'random':
        return random_split_data(origin_data, n_user)
    elif mode == 'iid':
        return iid_split_data(origin_data, n_user)
    elif mode == 'non_iid':
        return non_iid_split_data(origin_data, n_user)
    elif mode == 'diri':
        return diri_split_data(origin_data, n_user, alpha)

def random_split_data(origin_data, n_user):

    x, y = origin_data
    x, y = shuffle(x, y)

    total_num_data = len(x)
    num_client_data = total_num_data // n_user
    client_datasets = []
    s_idx = 0

    for i in range(n_user):
        e_idx = s_idx + num_client_data
        client_datasets.append({
            'x':x[s_idx: e_idx],
            'y':y[s_idx: e_idx]
             })
        s_idx = e_idx

    return client_datasets

def iid_split_data(origin_data, n_user):
    x, y = origin_data
    client_datasets = [{'x': [], 'y': []} for _ in range(n_user)]
    tmp_y = np.argmax(y, axis=1)
    unique_labels = np.unique(tmp_y, return_counts=False)
    user_idx_list = [[] for _ in range(n_user)]

    for label in unique_labels:
        label_indices = np.where(tmp_y == label)[0]
        data_per_user = len(label_indices) // n_user
        for i in range(n_user):
            user_samples = np.random.choice(label_indices, data_per_user, replace=False)
            label_indices = list(set(label_indices) - set(user_samples))
            user_idx_list[i].extend(user_samples)
    for i in range(n_user):
        client_datasets[i]['x'] = np.take(x, user_idx_list[i], axis=0)
        client_datasets[i]['y'] = np.take(y, user_idx_list[i], axis=0)

    return client_datasets


def non_iid_split_data(dataset, num_clients):
    np.random.seed(0)

    x, labels = dataset
    arg_labels = np.argmax(labels, axis=1)
    classes_per_client = 2
    client_datasets = [{'x': [], 'y': []} for _ in range(num_clients)]

    # data_loaders = [0] * num_clients
    num_shards = classes_per_client * num_clients
    num_imgs = int(len(x) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    #     no need to judge train ans test here
    idxs_labels = np.vstack((idxs, arg_labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)

    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
    for i in range(num_clients):
        client_datasets[i]['x'] = np.take(x, dict_users[i], axis=0)
        client_datasets[i]['y'] = np.take(labels, dict_users[i], axis=0)
    return client_datasets


def diri_split_data1(origin_data, n_user, alpha):

    x, labels = origin_data
    client_datasets = [{'x': [], 'y': []} for _ in range(n_user)]
    t_classes = len(np.unique(labels))
    t_idx_slice = [[] for _ in range(n_user)]

    for k in range(t_classes):
        t_idx_k = np.where(labels == k)[0]
        np.random.shuffle(t_idx_k)
        prop = np.random.dirichlet(np.repeat(alpha, n_user))
        t_prop = (np.cumsum(prop) * len(t_idx_k)).astype(int)[:-1]
        t_idx_slice = idx_slicer(t_idx_slice, t_idx_k, t_prop)

    for i in range(n_user):
        np.random.shuffle(t_idx_slice[i])

    for i in range(n_user):
        client_datasets[i]['x'] = np.take(x, t_idx_slice[i], axis=0)
        client_datasets[i]['y'] = np.take(labels, t_idx_slice[i], axis=0)

    return client_datasets

def diri_split_data(origin_data, n_user, alpha):
    x, labels = origin_data
    client_datasets = [{'x': [], 'y': []} for _ in range(n_user)]

    y_train_index = np.argmax(labels, axis=1)
    t_classes = len(np.unique(y_train_index))
    t_idx_slice = [[] for _ in range(n_user)]

    for k in range(t_classes):
        t_idx_k = np.where(y_train_index == k)[0]
        np.random.shuffle(t_idx_k)
        prop = np.random.dirichlet(np.repeat(alpha, n_user))
        t_prop = (np.cumsum(prop) * len(t_idx_k)).astype(int)[:-1]
        t_idx_slice = idx_slicer(t_idx_slice, t_idx_k, t_prop)

    for i in range(n_user):
        np.random.shuffle(t_idx_slice[i])

    for i in range(n_user):
        client_datasets[i]['x'] = np.take(x, t_idx_slice[i], axis=0)
        client_datasets[i]['y'] = np.take(labels, t_idx_slice[i], axis=0)

    return client_datasets

def idx_slicer(idx_slice, idx_k, prop):
    return [idx_j + idx.tolist() for idx_j, idx in zip(idx_slice, np.split(idx_k, prop))]