import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
import os
# from acconeer.exptool import a121
# import h5py
from sklearn.model_selection import train_test_split

# from sklearn.decomposition import PCA
# from tqdm import tqdm

import collections
import json
# import pickle as pk
from models import basemodel
# from matplotlib import animation 
# from preprocess import processing
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

RANDOM_SEED = 33

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def read_data(data_path, data_dict):
    """
    Read .npy data
    """
    for i in os.listdir(data_path):
        d = os.path.join(data_path, i)
        if os.path.isdir(d):
            for file in os.listdir(d):
                if file != '.DS_Store':
                    data_dict[i][file] = np.mean(np.load(os.path.join(d, file)), axis = 1)
    return data_dict

def createData(Dict : dict) -> dict :
    """
    Create data for plot
    """
    Data = dict()
    for i in Dict:
        temp = None
        for j in Dict[i]:
            if type(temp) == type(None):
                temp = Dict[i][j]
            else:
                temp = np.concatenate((temp, Dict[i][j]), axis = 0)
        Data[i] = temp
    return Data

def add_label(arr, label, label2idx_Dict):
        """
        Add labels to data
        """
        # label = list()
        # for i in idx2label_Dict:
        #     label.append(idx2label_Dict[i])
        label_list = [label2idx_Dict[label] for j in range(len(arr))]
        label_list = np.array(label_list)
        label_list = np.reshape(label_list, (len(arr), 1))
        # print(label_list.shape)
        labeled_arr = np.concatenate((arr, label_list), axis = 1)
        return labeled_arr

def concat(arr : list):
        temp = arr[0]
        for i in range(1, len(arr)):
            temp = np.concatenate((temp, arr[i]), axis = 0)
        return temp

def make_dataset(data : dict, classes, label2idx_Dict):
    """
    Make dataset for Machine learning
    """
    temp_arr = list()
    # temp_dict = dict()
    for i in data:
        # for i in data:
        temp_arr.append(add_label(data[i], i, label2idx_Dict))
    temp_data = concat(temp_arr)
    # Y = to_categorical(temp_data[:,-1], num_classes = len(label2idx_Dict))
    Y = to_categorical(temp_data[:,-1], num_classes = classes)
    X_train, X_test, Y_train, Y_test = train_test_split(temp_data[:,:-1], Y, random_state = RANDOM_SEED)

    return X_train, X_test, Y_train, Y_test
def train(
    model, X, Y,
    test_X, test_Y, 
    # callback,
    Epoch = 50,
    learning_rate = 1e-3,
    cp_path = None,
    ):
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = './model/' + cp_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only = True,
        save_weigths_only = False,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer = optimizer , loss = loss, 
                metrics = ['accuracy', 'categorical_crossentropy'])
    print('X shape : ', X.shape)
    model.build(input_shape = (1, X.shape[1]))          
    model.summary()
    history = model.fit(X, Y,  epochs = Epoch,
                validation_data = (test_X, test_Y),
                callbacks = [callback]
                )
    return history


def saveplot(history, plot_name):
    plot_name =  str(plot_name) + '.png'
    plt.subplot(4,1,1)
    plt.ylim(0,1)
    max = np.argmax(history['val_accuracy'])
    plt.plot(history['val_accuracy'])
    plt.plot(max, history['val_accuracy'][max], 'o', ms = 10, label=round(history['val_accuracy'][max],4))
    plt.title('val_accuracy')
    plt.legend(loc = 'center right',fontsize=15)

    plt.subplot(4,1,2)
    plt.ylim(0,1)
    min = np.argmin(history['val_loss'])
    plt.plot(history['val_loss'])
    plt.plot(min, history['val_loss'][max], 'o', ms = 10, label=round(history['val_loss'][max],4))
    plt.title('val_loss')
    plt.legend(loc = 'center right',fontsize=15)

    plt.subplot(4,1,3)
    plt.ylim(0,1)
    max = np.argmax(history['accuracy'])
    plt.plot(history['accuracy'])
    plt.plot(max, history['accuracy'][max], 'o', ms = 10, label=round(history['accuracy'][max],4))
    plt.title('accuracy')
    plt.legend(loc = 'center right',fontsize=15)

    plt.subplot(4,1,4)
    plt.ylim(0,1)
    min = np.argmin(history['loss'])
    plt.plot(history['loss'])
    plt.plot(min, history['loss'][max], 'o', ms = 10, label=round(history['loss'][max],4))
    plt.title('loss')
    plt.legend(fontsize=15)
    plt.legend(loc = 'center right',fontsize=15)
    createDirectory(os.path.dirname(plot_name))
    plt.savefig(plot_name, dpi = 300)

def main():

    npy_path = './counting_data_npy'

    labels = sorted(os.listdir(npy_path))
    labels.remove('rename.ipynb')

    # label load
    with open('label.json','r') as f:
        label2idx_Dict = json.load(f)

    idx2label_Dict = {v:k for k,v in label2idx_Dict.items()}
    CLASSES = len(label2idx_Dict)

    label = [i for i in label2idx_Dict]

    data_dict = collections.defaultdict(dict)
    npy_dict = read_data(npy_path, data_dict)
    # Data_dict = createData(npy_dict)
    initial_data = np.mean(npy_dict['s_00']['s_00.npy'], axis = 0)
    denoised_dict = collections.defaultdict(dict)
    for i in npy_dict:
        for j in npy_dict[i]:
            denoised_dict[i][j] = npy_dict[i][j] - initial_data
    Data_dict = createData(denoised_dict)
    
    X_train, X_test, Y_train, Y_test = make_dataset(Data_dict, CLASSES, label2idx_Dict)

    gpu = 2
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
            tf.config.experimental.set_virtual_device_configuration(
                gpus[gpu],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
        except RuntimeError as e:
            # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
            print(e)

    title = './result/denoised'
    model = basemodel(CLASSES)
    history = train(model, X_train, Y_train, X_test, Y_test, Epoch = 50, cp_path=title)
    saveplot(history.history, title)

if __name__ == '__main__':
    main()