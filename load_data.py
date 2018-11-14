from keras.datasets import cifar10
import numpy as np
from keras.utils import np_utils
import qrcode,cv2
nb_classes = 2
from PIL import Image
import matplotlib.pyplot as plt
import os

def cifar():
    print("Loading cifar10 data ...")

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    ## choose "airplane" and "horse" training pics
    airplane_idx = []
    horse_idx = []
    for index in range(len(y_train)):
        label = y_train[index]
        if label == 0:
            airplane_idx.append(index)
        elif label == 7:
            horse_idx.append(index)

    airplane_data = X_train[airplane_idx,:,:,:]
    horse_data = X_train[horse_idx,:,:,:]
    X_train = np.concatenate([airplane_data,horse_data],axis=0)
    airplane_data = np.zeros(shape=(5000,1))
    horse_data = np.ones(shape=(5000,1))
    y_train = np.concatenate([airplane_data,horse_data],axis=0)

    ## choose "airplane" and "horse" test pics
    airplane_idx = []
    horse_idx = []
    for index in range(len(y_test)):
        label = y_test[index]
        if label == 0:
            airplane_idx.append(index)
        elif label == 7:
            horse_idx.append(index)

    airplane_data = X_test[airplane_idx, :, :, :]
    horse_data = X_test[horse_idx, :, :, :]
    X_test = np.concatenate([airplane_data, horse_data], axis=0)
    airplane_data = np.zeros(shape=(1000,1))
    horse_data = np.ones(shape=(1000,1))
    y_test = np.concatenate([airplane_data, horse_data], axis=0)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    # Y_train = np_utils.to_categorical(y_train, nb_classes)
    # Y_test = np_utils.to_categorical(y_test, nb_classes)

    Y_train = y_train.reshape(-1,1)
    Y_test = y_test.reshape(-1,1)


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    print("Finished data loading!!!")
    return X_train,Y_train,X_test,Y_test

def load_cifar10():
    X_train_public, Y_train_public, X_test_public, Y_test_public = cifar()
    X_train_secret,X_test_secret = [],[]
    X_train_public_temp,X_test_public_temp = [],[]


    print("Generating secret training dataset ...")
    for index,pic in enumerate(X_train_public):
        pic_temp = (pic-127.5) / 127.5
        X_train_public_temp.append(pic_temp)
        qr = gene_qr(index)
        qr_pos = np.random.uniform(0,21,size=2)
        pic = Image.fromarray(np.uint8(pic))
        pic.paste(qr,(int(qr_pos[0]),int(qr_pos[1])))
        # plt.imshow(pic)
        # plt.show()
        pic = np.asarray(pic)
        pic = (pic.copy()-127.5) / 127.5
        X_train_secret.append(pic)
        if index % 100 == 0:
            print(index,"/",len(X_train_public))


    print("Generating secret test dataset ...")
    for index,pic in enumerate(X_test_public):
        pic_temp = (pic - 127.5) / 127.5
        X_test_public_temp.append(pic_temp)
        qr = gene_qr(index)
        qr_pos = np.random.uniform(0,21,size=2)
        pic = Image.fromarray(np.uint8(pic))
        pic.paste(qr, (int(qr_pos[0]), int(qr_pos[1])))
        # plt.imshow(pic)
        # plt.show()
        pic = np.asarray(pic)
        pic = (pic.copy() - 127.5) / 127.5
        X_test_secret.append(pic)
        if index % 100 == 0:
            print(index,"/",len(X_test_public))

    X_train_public = np.array(X_train_public_temp)
    X_test_public = np.array(X_test_public_temp)
    X_train_secret = np.array(X_train_secret)
    X_test_secret = np.array(X_test_secret)

    return X_train_public, Y_train_public, X_test_public, Y_test_public,\
           X_train_secret, Y_train_public, X_test_secret, Y_test_public


def gene_qr(index):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=2,
    )
    qr.add_data(str(index))
    qr.make(fit=True)
    img = qr.make_image()
    img = img.resize((10,10))
    return img
