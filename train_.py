from __future__ import print_function, division

import load_data
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import Conv2D, Deconv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import random
import matplotlib.pyplot as plt
import os,time
from PIL import Image
from tqdm import tqdm
import numpy as np

class ae_gan():
    def __init__(self):
        self.img_rows, self.img_cols, self.channels = 32, 32, 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(lr=0.0002,beta_1=0.5)
        self.attack = self.discriminator()
        self.attack.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.ae = self.autoencoder()
        input_image = Input(shape=self.img_shape)
        generated_img = self.ae(input_image)

        self.attack.trainable = False
        valid = self.attack(generated_img)

        self.combined_model = Model(input_image,[valid,generated_img])
        self.combined_model.compile(loss=['binary_crossentropy','mae'],
                                    loss_weights=[0.5, 0.5],
                                    optimizer=optimizer)

    def autoencoder(self):
        input = Input(shape=self.img_shape)
        h = Conv2D(64, (5, 5), strides=2, padding='same', activation='relu')(input)
        h = Conv2D(128, (5, 5), strides=2, padding='same', activation='relu')(h)
        h = Conv2D(256, (5, 5), strides=2, padding='same', activation='relu')(h)
        encoded = Conv2D(512, (5, 5), strides=2, padding='same', activation='relu')(h)

        h = Deconv2D(512, (5, 5), strides=2, padding='same', activation='relu')(encoded)
        h = Deconv2D(256, (5, 5), strides=2, padding='same', activation='relu')(h)
        h = Deconv2D(128, (5, 5), strides=2, padding='same', activation='relu')(h)
        decoded = Deconv2D(3, (5, 5), strides=2, padding='same', activation='tanh')(h)

        auto_encoder = Model(input, decoded)
        auto_encoder.summary()

        return auto_encoder

    def discriminator(self):
        input = Input(shape=self.img_shape)
        h = Conv2D(64, (5, 5), strides=2, padding='same', activation='relu')(input)
        h = Conv2D(128, (5, 5), strides=2, padding='same', activation='relu')(h)
        h = Conv2D(256, (5, 5), strides=2, padding='same', activation='relu')(h)
        h = Conv2D(512, (5, 5), strides=2, padding='same', activation='relu')(h)
        h = Flatten()(h)
        output_secret = Dense(1, activation='relu')(h)
        discriminator = Model(input, output_secret)
        discriminator.summary()

        return discriminator

    def train(self,epochs, batch_size=128, sample_interval=50):
        x_train_public, y_train_public, _, _, \
        x_train_secret, y_train_secret, _, _ = load_data.load_cifar10()

        label_secret = np.ones(shape=(batch_size, 1))
        label_public = np.zeros(shape=(batch_size, 1))

        for epoch in range(1,epochs+1):
            start = time.time()
            print("In the epoch ",epoch,"/",epochs)

            ####### generate pics for public pics #######
            idx_public = random.sample(range(0, x_train_public.shape[0]), batch_size)
            image_batch_public = x_train_public[idx_public, :, :, :]
            generated_images_public = self.ae.predict(image_batch_public)

            ####### generate pics for secret pics #######
            idx_secret = random.sample(range(0, x_train_secret.shape[0]), batch_size)
            image_batch_secret = x_train_secret[idx_secret, :, :, :]
            generated_images_secret = self.ae.predict(image_batch_secret)

            l1 = self.attack.train_on_batch(image_batch_public,label_public)
            l2 = self.attack.train_on_batch(generated_images_public,label_public)
            l3 = self.attack.train_on_batch(image_batch_secret,label_secret)
            l4 = self.attack.train_on_batch(generated_images_secret,label_secret)

            g_loss1 = self.combined_model.train_on_batch(image_batch_public,[label_public,image_batch_public])
            g_loss2 = self.combined_model.train_on_batch(image_batch_secret,[label_public,image_batch_secret])

            print("Epoch ",epoch,"took time",time.time()-start)
            if epoch % 20 == 0:
                self.save_model(epoch)
                self.sample_images(image_batch_secret[0],epoch,'secret')
                self.sample_images(image_batch_public[0],epoch,'public')
    def sample_images(self, image, epoch, label):
        image = np.expand_dims(image,axis=0)
        gen_imgs = self.ae.predict(image) # output pixel size is between (-1,1)

        gen_imgs = 127.5 * gen_imgs + 127.5

        data = gen_imgs[0].astype(np.uint8)
        output_path = './images_vaegan_'+label+'/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        img = Image.fromarray(data,'RGB')
        img.save(output_path + "%d.png" % epoch)
        plt.close()

    def save_model(self, epoch):

        def save(model, epoch, model_name):
            output_path = './models_vaegan/'
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            model_path = output_path + str(epoch) + "_" + model_name + ".h5"
            model.save(model_path)

        save(self.ae, epoch, "autoencoder")
        # save(self.attack, epoch, "discriminator")

if __name__ == '__main__':
    model = ae_gan()
    model.train(epochs=1000,batch_size=32,sample_interval=200)


