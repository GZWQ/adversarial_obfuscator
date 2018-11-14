from __future__ import print_function
from keras.layers import Dense, Dropout, Flatten,Input
from keras.optimizers import Adam,SGD
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
import load_data
from keras.models import load_model
import numpy as np
import os

def generated_images(data):
    model = load_model('./models_vaegan/1980_autoencoder.h5')
    generated_images = model.predict(data)
    return generated_images

x_train_public, y_train_public, x_test_public, y_test_public,\
           x_train_secret, y_train_secret, x_test_secret, y_test_secret  = load_data.load_cifar10()
x_train_public_generated = generated_images(x_train_public)
x_test_public_generated = generated_images(x_test_public)
x_train_secret_generated = generated_images(x_train_secret)
x_test_secret_generated = generated_images(x_test_secret)


def cnn_model():
    d0 = Input((x_train_public.shape[1:]))
    # x0 = Dense(img_rows*img_cols*1, activation = 'relu')(d0)
    # x0 = Reshape((img_rows,img_cols,1))(x0)
    x = Conv2D(32, (5, 5), padding='same', name='id_conv1')(d0)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)

    x = Conv2D(32, (5, 5), padding='same', name='id_conv2')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)

    x = Conv2D(32, (3, 3), padding='same', name='id_conv3')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)

    x = Conv2D(32, (3, 3), padding='same', name='id_conv4')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(32, name='id_dense1')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    output_main = Dense(1, activation='sigmoid', name='id_dense2')(x)

    return Model(d0, output_main)


sgd = SGD(lr=0.005, momentum=0.9, decay=1e-7, nesterov=True)
model = cnn_model()
model.compile(loss=['binary_crossentropy'],
              optimizer=sgd,
              metrics=['accuracy'])
model.summary()

def eval_fun(x_train,y_train,x_test,y_test,util_privacy):
    output = '../'+util_privacy[0]+'/'
    if not os.path.isdir(output):
        os.mkdir(output)
    result = model.fit(x_train,y_train,
                                batch_size=50,
                                epochs=10,
                                validation_data=(x_test,y_test),
                                shuffle=True)
    filename = output + util_privacy[1]+ '.txt'
    print(result.history['val_acc'])
    with open(filename, mode='w') as f1:
        f1.write(str(result.history['val_acc']))
        f1.write('\n')
        f1.write('Ave_ACC is : ' + str
        (sum(result.history['val_acc']) / len(result.history['val_acc'])))


if __name__ == '__main__':


    #######################################################

    original_data_train = np.concatenate((x_train_public, x_train_secret), axis=0)
    original_data_train_class_label = np.concatenate((y_train_public,y_train_secret),axis=0)
    original_data_test = np.concatenate((x_test_public,x_test_secret),axis=0)
    original_data_test_class_label = np.concatenate((y_test_public,y_test_secret),axis=0)

    generated_data_train = np.concatenate((x_train_public_generated, x_train_secret_generated), axis=0)
    generated_data_train_class_label = np.concatenate((y_train_public,y_train_secret),axis=0)
    generated_data_test = np.concatenate((x_test_public_generated, x_test_secret_generated), axis=0)
    generated_data_test_class_label = np.concatenate((y_test_public,y_test_secret),axis=0)

    label_original_data = np.zeros(shape=(len(generated_data_train), 1))
    label_original_data[len(generated_data_train) // 2:, :] = 1

    with_secret_label = np.zeros(shape=(len(generated_data_test), 1))
    with_secret_label[len(generated_data_test) // 2:, :] = 1
    #######################################################
    Eval = 'up' # 'privacy' / 'up'
    if Eval == 'util':
        print("In the UTILITY process")

        #eval_fun(original_data_train, original_data_train_class_label,
         #                          original_data_test, original_data_test_class_label,['utility','baseline'])


        eval_fun(original_data_train, original_data_train_class_label,
                            generated_data_test, generated_data_test_class_label,['utility','a'])


        eval_fun(generated_data_train, generated_data_train_class_label,
                            generated_data_test, generated_data_test_class_label,['utility','b'])

    elif Eval == 'privacy':
        
        print("In the PRIVACY process") 
       
        eval_fun(original_data_train, label_original_data,
                                   generated_data_test, with_secret_label,['privacy','weak'])

        eval_fun(generated_data_train, label_original_data,
                                    generated_data_test, with_secret_label,['privacy','strong'])
    elif Eval =='up':
        print("In the BOTH process")
        #eval_fun(original_data_train, original_data_train_class_label,
         #        original_data_test, original_data_test_class_label, ['utility', 'baseline'])

        eval_fun(original_data_train, original_data_train_class_label,
                 generated_data_test, generated_data_test_class_label, ['utility', 'a'])

        eval_fun(generated_data_train, generated_data_train_class_label,
                 generated_data_test, generated_data_test_class_label, ['utility', 'b'])

        eval_fun(original_data_train, label_original_data,
                 generated_data_test, with_secret_label, ['privacy', 'weak'])

        eval_fun(generated_data_train, label_original_data,
                 generated_data_test, with_secret_label, ['privacy', 'strong'])


