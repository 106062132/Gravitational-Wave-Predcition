import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import os
import shutil
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.utils import np_utilsy
from keras import backend as K
import os
import gc
from PIL import Image
import numpy as np
import seaborn as sn
import pandas as pd
import datetime
import itertools
import matplotlib.pyplot as plt
import pickle
import random
import sys


################################################################################
def list_fail_case(f):
    '''
    Create a list that contains all the failure cases (ie. bounce(s)=-1 ??).
    f: h5 file.
    return:
    fail_list: list.
    not_fail_list: list.
    '''
    fail_num = []
    index = 0
    for item in f['reduced_data']['tbounce(s)']:
        if (item == -1):
            fail_num.append(index)
        index += 1

    fail_case = []
    for index in fail_num:
        fail_case.append([f['reduced_data']['A(km)'][index], f['reduced_data']['omega_0(rad|s)'][index],
                          f['reduced_data']['EOS'][index]])

    fail_list = []
    for item in fail_case:
        tmp = str(item[2]).split("b'")[1].split("'")[0]
        tmp = "A" + str(int(item[0])) + "w" + str(item[1]) + "0_" + tmp
        fail_list.append(tmp)

    non_fail_list = []
    for item in f['waveforms']:
        if (item not in fail_list):
            non_fail_list.append(item)

    return fail_list, non_fail_list


################################################################################
def create_y(non_fail_list, y):
    '''
    Create labels.
    non_fail_list: list.
    y: str. Use either 'w', 'EOS', or 'A'.
    return: array.
    '''
    labels = []
    index = 0
    for item in non_fail_list:
        if y == 'w':
            labels.append(float(item.split('_')[0].split('w')[1]))
        elif y == 'EOS':
            labels.append(str(item.split('_')[1]))
        elif y == 'A':
            labels.append(str(item.split('_')[0].split('w')[0].split('A')[1]))
        else:
            sys.exit("\n Use either 'w', 'EOS', or 'A' as the input. Check create_y().")
        index += 1

    # turn the type of labels into array.
    labels = np.array(labels)

    return labels


################################################################################
def create_x_image(f, non_fail_list, output_folder='./data/ftr', time_range=[-0.01, 0.006],
                    resolution={'figsize': (4, 4), 'dpi': 64}, ftype='jpeg', overwrite=False):
    '''
    Create features (image).
    f: h5 file.
    non_fail_list: list.
    output_folder: str.
    time_range: list.
    resolution: dict. Default=256*256.
    ftype: str. The file type.
    overwrite: boolean. whether to overwrite the exist files.
    return
    '''
    # create output folder
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    # create output file name
    w = resolution['figsize'][0] * resolution['dpi']
    h = resolution['figsize'][1] * resolution['dpi']
    fname0 = "img%s*%s_%sto%s" % (str(w), str(h), str(time_range[0]), str(time_range[1]))

    # draw the image which is 256*256.
    index = 0
    for item in f['waveforms']:
        if (item in non_fail_list):
            # save path
            fname = "%s/%s_%s.%s" % (output_folder, str(index), fname0, ftype)

            # if the file is already exist and needs no overwrite, then don't recreate the ftr again
            if all([os.path.exists(fname), not overwrite]):
                index += 1
                continue

            # create x image
            # for j in range(5):
            x = f['waveforms'][item]['t-tb(s)']
            y = f['waveforms'][item]['strain*dist(cm)']
            y = np.array(y)
            x = np.array(x)

            plt.figure(figsize=resolution['figsize'], dpi=resolution['dpi'])
            plt.plot(x, y / 3.08567758e22 * 1e21)  # why divided by this number?
            plt.xlim(time_range[0], time_range[1])
            plt.ylim(-20, 20)  # why use this numbers?
            plt.axis('off')

            plt.savefig(fname, dpi=resolution['dpi'])
        index += 1

    return


################################################################################
def create_x_strain(f, non_fail_list, output_folder='./data/ftr', time_range=[-0.01, 0.006], overwrite=False):
    '''
    Create features (time and strain distance).
    f: h5 file.
    non_fail_list: list.
    output_folder: str.
    time_range: list.
    overwrite: boolean. whether to overwrite the exist files.
    return
    '''
    # create output folder
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    # create output file name
    fname0 = "strain_%sto%s" % (str(time_range[0]), str(time_range[1]))

    # create the time & strain DF
    index = 0
    for item in f['waveforms']:
        if (item in non_fail_list):
            # save path
            fname = "%s/%s_%s.csv" % (output_folder, str(index), fname0)

            # if the file is already exist and needs no overwrite, then don't recreate the ftr again
            if all([os.path.exists(fname), not overwrite]):
                index += 1
                continue

            # create DF
            x = f['waveforms'][item]['t-tb(s)']
            y = f['waveforms'][item]['strain*dist(cm)']
            df = pd.DataFrame({'t': np.array(x), 'strain': np.array(y)})

            df.to_csv(fname)

        index += 1

    return


################################################################################
def load_x_image(f, non_fail_list, input_folder='./data/ftr', time_range=[-0.01, 0.006],
                 resolution={'figsize': (4, 4), 'dpi': 64}, ftype='jpeg', turn_array=True):
    '''
    Load in features (image), and turn images into array or not.
    f: h5 file.
    non_fail_list: list.
    input_folder: str.
    time_range: list.
    resolution: dict. Default=256*256.
    ftype: str. The file type.
    turn_array: boolean. whether to turn a list of images into arrays.
    return: list/array. The a list of images or image arrays.
    '''
    # create input file name
    w = resolution['figsize'][0] * resolution['dpi']
    h = resolution['figsize'][1] * resolution['dpi']
    fname0 = "img%s*%s_%sto%s" % (str(w), str(h), str(time_range[0]), str(time_range[1]))

    # load x images
    data = []
    index = 0
    for item in f['waveforms']:
        if (item in non_fail_list):
            # load path
            fname = "%s/%s_%s.%s" % (input_folder, str(index), fname0, ftype)
            image = Image.open(fname).convert('L')
            data.append(np.array(image))
        index += 1

    # transform the data type to numpy array
    if turn_array:
        data = np.array(data)

    return data


################################################################################
def label_encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)


################################################################################
def label_decode(le, one_hot_label):
    dec = np.argmax(one_hot_label, axis=1)
    return le.inverse_transform(dec)


################################################################################
def prepare_NN(model_name, num_of_label):
    if model_name == 'cnn1':
        # our model
        model = Sequential()
        model.add(Conv2D(64, (5, 5), input_shape=(256, 256, 1), padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (5, 5), padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (5, 5), padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_of_label, activation='softmax'))
        # model.add(Dense(1, activation='relu'))
        optimizer = keras.optimizers.Adam(learning_rate=0.00001)
        # optimizer = keras.optimizers.Adam(learning_rate=0.000003)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        # model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    return model


################################################################################
def evaluate_model(y_test, X_test, label_encoder=None):
    if label_encoder is None:
        # check the accuracy.
        y_pred = model.predict(X_test)
        index = 0
        correct = 0
        for item in y_pred:
            if (abs(item - y_test[index]) <= 0.25):
                correct += 1
            index += 1
        acc = correct / index
    else:
        # check the accuracy.
        answer = np.argmax(model.predict(X_test), axis=1)
        answer_onehot = np.zeros((answer.size, answer.max() + 1))
        answer_onehot[np.arange(answer.size), answer] = 1
        y_pred = label_decode(label_encoder, answer_onehot)
        y_test = label_decode(label_encoder, y_test)
        index = 0
        correct = 0
        for item in y_pred:
            if (item == y_test[index]):
                correct += 1
            index += 1
        acc = correct / index
    return y_test, y_pred, acc


################################################################################
def plot_confusion_matrix(labels, y_test, y_pred, save_path=None):
    # plot confusion matrix
    y_pred = y_pred.astype('str')
    y_test = np.array(y_test).astype('str')

    cm = confusion_matrix(y_test, y_pred)

    tmp = []
    for i in range(len(np.unique(labels))):
        tmp.append(0.5 * i)
    df_cm = pd.DataFrame(cm, index=[i for i in tmp],
                         columns=[i for i in tmp])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)

    if save_path is not None:
        plt.savefig(save_path)
    return


################################################################################
def list_sampling(lst, n=None, pct=None, with_replacement=False, sampling_times=10, shuffle_list=True, cover_all=True):
    '''
    :param lst: target list
    :param n: separate lst into n groups
    :param pct: separate lst into (1/pct) groups with (len(lst)*pct) in every group
    pct=0.672 -> test_set=154, train+cv=60+15
    pct=0.562 -> test_set=129, train+cv=80+20
    pct=0.452 -> test_set=104, train+cv=100+25
    pct=0.345 -> test_set=79, train+cv=120+30
    pct=0.126 -> test_set=29, train+cv=160+40
    '''

    if shuffle_list:
        random.shuffle(lst)

    sample_set = []
    remain_set = []
    if with_replacement:
        if all([n is None, pct > 0, pct < 1]):
            num = round(pct * len(lst))

            if cover_all:
                # make sure every element has been selected once
                if pct * sampling_times < 1:
                    sys.exit("\n 'sampling_times' is too small to 'cover_all_data'. Check list_sampling().")

                remain_times = sampling_times
                while True:
                    # evenly divide the target list into lists
                    random.shuffle(lst)
                    _sampling_lst = [lst[i:i + num] for i in range(0, len(lst), num)]
                    _l = _sampling_lst[-2] + _sampling_lst[-1]
                    _l = _l[-num:]
                    _sampling_lst = _sampling_lst[:-1]
                    _sampling_lst.append(_l)

                    # update sampling_lst and remain_time
                    if remain_times >= len(_sampling_lst):
                        remain_times -= len(_sampling_lst)
                        sample_set += _sampling_lst
                        if remain_times == 0:
                            break
                    else:
                        _sampling_lst = _sampling_lst[:remain_times]
                        sample_set += _sampling_lst
                        break


            else:
                sample_set = []
                for i in range(sampling_times):
                    x = random.sample(lst, num)
                    sample_set.append(x)

        else:
            sys.exit("\n 'with_replacement' and 'sampling_times' can be used only with 'Percent'(pct=0.1, 0.2, 0.3,...). Check list_sampling().")



    else:
        if all([n is not None, pct is None]):
            if n > len(lst):
                sys.exit("\n n > len(lst)! Check list_sampling().")
            else:
                division = len(lst) / n
        elif all([n is None, pct > 0, pct < 1]):
            val = 1 / pct
            n = round(val)
            division = len(lst) / n
        else:
            sys.exit("\n Use either Number(n=1, 2, 3...) or Percent(pct=0.1, 0.2, 0.3,...) to separate the list. Check list_sampling().")

        sample_set = [lst[round(division * i):round(division * (i + 1))] for i in range(n)]

    # remain_set
    for i in range(len(sample_set)):
        _lst = list(set(lst) - set(sample_set[i]))
        remain_set.append(_lst)

    return sample_set, remain_set

################################################################################