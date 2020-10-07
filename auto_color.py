# -*- coding: utf-8 -*-
import numpy as np
from keras.models import load_model, Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate
from keras.optimizers import Adam
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image
import tensorflow as tf

from config import Env

def getModel(img_height, img_width):
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
    # model.summary()

    return model

img_height, img_width = 90, 160
model = getModel(img_height, img_width)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.5), metrics=['accuracy'])
model.load_weights(Env.MODEL_PATH)
model._make_predict_function()
graph = tf.get_default_graph()

def autoColor(pil_image):
    width, height = pil_image.size
    pil_image = pil_image.resize((img_width, img_height)).convert('L')
    X, Y = [], []
    # 単独着彩
    X.append(img_to_array(pil_image))
    X = np.asarray(X)
    X = X.astype(np.float32) / 255
    global graph
    with graph.as_default():
        image = (model.predict(np.reshape(X[0], (1, img_height, img_width, 1)), verbose=0))
    image = np.reshape(image, (img_height, img_width, 3))
    image = image * 255
    result_image = Image.fromarray(image.astype(np.uint8))
    resizeImage = result_image.resize((width, height))
    return resizeImage