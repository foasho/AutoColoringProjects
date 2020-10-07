from keras.models import Sequential, Model
from keras.layers import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys
from keras.datasets import mnist
from keras.optimizers import Adam
from PIL import Image
import math
import numpy as np
import keras.backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import re
from PIL import Image

import tensorflow as tf
import six

session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=False, visible_device_list=""))
tf.Session(config=session_config)
six.moves.input()

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]


def unet(img_height, img_width):
    input_layer = Input(shape=(img_height, img_width, 1))
    conv1 = Conv2D(32, (3, 3), strides=(1, 1))(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(64, (4, 4), strides=(2, 2))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv3 = Conv2D(128, (3, 4), strides=(2, 2))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    conv4 = Conv2D(256, (3, 4), strides=(2, 2))(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)

    conv5 = Conv2D(512, (2, 2), strides=(1, 1))(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    # 底辺
    conv6 = Conv2D(1024, (2, 2), strides=(1, 1))(conv5)  # 9x17 -> 8x16
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    # ##############

    # 折返し
    uconv6 = Conv2DTranspose(512, (2, 2), strides=(1, 1))(conv6)
    uconv6 = BatchNormalization()(uconv6)
    uconv6 = Activation('relu')(uconv6)
    uconv6 = Concatenate()([conv5, uconv6])

    uconv5 = Conv2DTranspose(256, (2, 2), strides=(1, 1))(uconv6)
    uconv5 = BatchNormalization()(uconv5)
    uconv5 = Activation('relu')(uconv5)
    uconv5 = Concatenate()([conv4, uconv5])

    uconv4 = Conv2DTranspose(128, (3, 4), strides=(2, 2))(uconv5)
    uconv4 = BatchNormalization()(uconv4)
    uconv4 = Activation('relu')(uconv4)
    uconv4 = Concatenate()([conv3, uconv4])

    uconv3 = Conv2DTranspose(64, (3, 4), strides=(2, 2))(uconv4)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Activation('relu')(uconv3)
    uconv3 = Concatenate()([conv2, uconv3])

    uconv2 = Conv2DTranspose(32, (4, 4), strides=(2, 2))(uconv3)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = Activation('relu')(uconv2)
    uconv2 = Concatenate()([conv1, uconv2])

    uconv1 = Conv2DTranspose(3, (3, 3), strides=(1, 1))(uconv2)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Activation('relu')(uconv1)

    model = Model(input_layer, uconv1)
    model.summary()

    return model


def train(x_dir, y_dir, img_height, img_width, Batch_size, Epoch_num, stepping_num, backup_num, model_dir, model_name):
    X, Y = [], []
    print("read x")
    for img in list_pictures(x_dir):
        img = img_to_array(load_img(img, target_size=(img_height, img_width), color_mode='grayscale'))
        img = np.reshape(img, (img_height, img_width, 1))
        X.append(img)
    print("read complete x")

    print("read y")
    for img in list_pictures(y_dir):
        img = img_to_array(load_img(img, target_size=(img_height, img_width)))
        Y.append(img)
    print("read complete y")

    X = np.asarray(X)
    X = X.astype(np.float32) / 255
    Y = np.asarray(Y)
    Y = Y.astype(np.float32) / 255
    model = unet(img_height, img_width)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5, beta_1=0.1))

    num_batches = int(X.shape[0] / Batch_size)
    print('Number of batches:', num_batches)
    stepping_epoch = Epoch_num//stepping_num
    print("GenerateImage Step OutPut Epoch:", stepping_epoch)
    backup_epoch = Epoch_num//backup_num
    print("WeightsData Step OutPut Epoch:", backup_epoch)
    for epoch in range(Epoch_num):
        for index in range(num_batches):
            X_batch = X[index * Batch_size:(index + 1) * Batch_size]
            Y_batch = Y[index * Batch_size:(index + 1) * Batch_size]

            # # 生成画像を出力
            if index % stepping_epoch == 0:
                image = (model.predict(np.reshape(X[epoch * 9 % len(X)], (1, img_height, img_width, 1)), verbose=0))
                image = np.reshape(image, (img_height, img_width, 3))
                image = image * 255
                if not os.path.exists(GENERATED_IMAGE_PATH):
                    os.makedirs(GENERATED_IMAGE_PATH)
                Image.fromarray(image.astype(np.uint8)).save(GENERATED_IMAGE_PATH + "%04d_%04d.png" % (epoch, index))

            if epoch % backup_epoch == 999:
                model.save_weights(model_dir+str(epoch) + '_' + model_name)

            u_loss = model.train_on_batch(X_batch, Y_batch)
            print("epoch: %d, batch: %d, u_loss: %f" % (epoch, index, u_loss))

    model.save_weights(model_dir+model_name)

def predict(img_height, img_width, target_img_path, model_path):
    model = unet(img_height, img_width)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.5), metrics=['accuracy'])
    model.load_weights(model_path)
    X, Y = [], []

    # 単独着彩
    X.append(img_to_array(load_img(target_img_path, target_size=(img_height, img_width), color_mode='grayscale')))

    X = np.asarray(X)
    X = X.astype(np.float32) / 255
    image = (model.predict(np.reshape(X[0], (1, img_height, img_width, 1)), verbose=0))
    image = np.reshape(image, (img_height, img_width, 3))
    image = image * 255
    pil_image = Image.fromarray(image.astype(np.uint8))
    width, height = Image.open(target_img_path).size
    resizeImage = pil_image.resize((width, height))
    resizeImage.save("predict.png")

if __name__=="__main__":
    Batch_size = 16
    Epoch_num = 10000
    stepping_num = 10#学習中モデルを使って画像を生成する回数
    backup_num = 3#学習中にモデルを何回保存するか
    img_height, img_width = 90, 160#PCが計算に耐えられるなら大きければ大きいほどいい
    GENERATED_IMAGE_PATH = './images/generated_images/'  # 生成画像の保存先ディレクトリ
    model_dir = "./model/"
    model_name = "AutoColor.h5"
    x_dir = './images/edge/'
    y_dir = './images/color/'

    #モデル生成　※評価だけしたい場合はコメントアウト
    train(x_dir, y_dir, img_height, img_width, Batch_size, Epoch_num, stepping_num, backup_num, model_dir, model_name)

    #評価
    target_img_path = "./images/example/test_predict.jpg"
    predict(img_height, img_width, target_img_path, model_dir+model_name)#評価